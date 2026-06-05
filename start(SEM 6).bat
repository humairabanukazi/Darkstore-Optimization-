@echo off
echo Starting Dark Store OS...
echo.

echo Starting Flask API...
start cmd /k "cd /d C:\Users\Humaira\Downloads\darkstore_project\python_model && python app.py"

timeout /t 3

echo Starting Node.js Server...
start cmd /k "cd /d C:\Users\Humaira\Downloads\darkstore_project\node_app && node server.js"

timeout /t 2

echo Opening Browser...
start http://localhost:3000

echo Done! Dark Store OS is running!



