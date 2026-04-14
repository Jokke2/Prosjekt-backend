Starte backend: python -m uvicorn main:app --reload
Test backend i annen terminal: Invoke-RestMethod -Method Post -Uri "http://localhost:8000/weather" -ContentType "application/json" -Body '{"latitude": 59.91, "longitude": 10.75}'

