import os
import json
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
# ── ENDRET: AsyncAnthropic ──
from anthropic import AsyncAnthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv()

app = FastAPI()
# ── ENDRET: async klient ──
anthropic_client = AsyncAnthropic(api_key=os.getenv("CLAUDE_API_KEY"))
BACKEND_KEY = os.getenv("BACKEND_KEY")

MODEL = "claude-haiku-4-5-20251001"
MET_MCP_URL = "https://webapi.met.no/mcp-server"

cached_tools = None


async def get_tools():
    global cached_tools
    if cached_tools is not None:
        return cached_tools

    async with streamable_http_client(MET_MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            cached_tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools_result.tools
            ]
            return cached_tools


def trim_weather_data(raw_text: str) -> str:
    try:
        data = json.loads(raw_text)
        forecast_text = data.get("forecast", raw_text)
        sections = forecast_text.split("\n## ")

        if len(sections) >= 2:
            header = sections[0]
            first_time = sections[1]
            return header + "\n## " + first_time

        return raw_text
    except (json.JSONDecodeError, AttributeError):
        return raw_text[:500]


# ── NYTT: robust JSON-parsing som håndterer markdown og tomme svar ──
def extract_json(text: str) -> dict:
    """Forsøker å parse JSON fra Claude-svar, håndterer markdown og edge cases."""
    text = text.strip()

    # Fjern markdown code blocks
    if "```" in text:
        # Finn innholdet mellom første og siste ```
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            # Fjern eventuell "json" språk-tag
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{"):
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue

    # Forsøk direkte parsing
    if text.startswith("{"):
        return json.loads(text)

    # Forsøk å finne JSON-objekt inne i teksten
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])

    raise ValueError(f"Kunne ikke finne JSON i svar: {text[:200]}")


class AuroraRequest(BaseModel):
    latitude: float
    longitude: float
    aurora_score: float
    k_index: float


class AuroraResponse(BaseModel):
    cloud_cover_percent: int
    aurora_probability_percent: int
    description: str


@app.post("/aurora")
async def get_aurora(
    request: AuroraRequest,
    x_api_key: str = Header(...)
):
    if x_api_key != BACKEND_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        tools = await get_tools()

        async with streamable_http_client(MET_MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                user_message = (
                    f"I need to assess northern lights visibility at "
                    f"{request.latitude},{request.longitude}.\n\n"
                    f"Additional data I already have:\n"
                    f"- Aurora activity score: {request.aurora_score}\n"
                    f"- Kp-index: {request.k_index}\n\n"
                    f"Please fetch BOTH cloud cover AND sunrise/sunset data "
                    f"for this location to determine if it is currently dark. "
                    f"Northern lights are only visible when it is dark. "
                    f"Use this as a factor in your probability estimate.\n\n"
                    f"Combine ALL factors (darkness, cloud cover, aurora score, "
                    f"kp-index, and latitude) to estimate the probability "
                    f"of actually seeing the northern lights.\n\n"
                    f"Respond with JSON only, no markdown:\n"
                    f'{{"cloud_cover_percent": N, '
                    f'"aurora_probability_percent": N, '
                    f'"description": "A short paragraph explaining how '
                    f'you arrived at this probability, considering all factors."}}'
                )

                messages = [{"role": "user", "content": user_message}]

                # ── ENDRET: maks 10 tool-loops for å unngå uendelig loop ──
                max_iterations = 10

                for i in range(max_iterations):
                    # ── ENDRET: await pga AsyncAnthropic ──
                    response = await anthropic_client.messages.create(
                        model=MODEL,
                        # ── ENDRET: økt fra 300 til 1024 for å unngå avkuttet svar ──
                        max_tokens=1024,
                        tools=tools,
                        messages=messages,
                    )

                    if response.stop_reason == "tool_use":
                        tool_results = []

                        for block in response.content:
                            if block.type == "tool_use":
                                result = await session.call_tool(
                                    block.name,
                                    arguments=block.input,
                                )

                                trimmed_text = trim_weather_data(
                                    result.content[0].text
                                )

                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": trimmed_text,
                                })

                        messages.append({
                            "role": "assistant",
                            "content": response.content,
                        })
                        messages.append({
                            "role": "user",
                            "content": tool_results,
                        })

                    else:
                        # ── ENDRET: samle tekst fra alle blokker ──
                        final_text = ""
                        for block in response.content:
                            if hasattr(block, "text"):
                                final_text += block.text

                        # ── NYTT: logg hva Claude faktisk svarte ──
                        print(f"[DEBUG] Claude raw response: '{final_text[:500]}'")
                        print(f"[DEBUG] Stop reason: {response.stop_reason}")

                        # ── NYTT: håndter tomt svar ──
                        if not final_text.strip():
                            print("[WARN] Claude returnerte tomt svar, ber om JSON på nytt")
                            messages.append({
                                "role": "assistant",
                                "content": response.content,
                            })
                            messages.append({
                                "role": "user",
                                "content": (
                                    "You didn't return any text. "
                                    "Please respond with ONLY the JSON object, nothing else:\n"
                                    '{"cloud_cover_percent": N, '
                                    '"aurora_probability_percent": N, '
                                    '"description": "..."}'
                                ),
                            })
                            # Fortsett loopen for å prøve igjen
                            continue

                        # ── ENDRET: bruk robust JSON-parser ──
                        data = extract_json(final_text)

                        return AuroraResponse(
                            cloud_cover_percent=int(
                                data["cloud_cover_percent"]
                            ),
                            aurora_probability_percent=int(
                                data["aurora_probability_percent"]
                            ),
                            description=data["description"],
                        )

                # ── NYTT: hvis vi brukte alle iterasjoner uten svar ──
                raise HTTPException(
                    500,
                    "Claude klarte ikke returnere gyldig JSON etter maks forsøk"
                )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)