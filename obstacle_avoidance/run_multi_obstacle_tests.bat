@echo off
REM Batch script to run multi-obstacle tests
REM This will test with 5, 10, 15, and 20 obstacles automatically

echo ======================================================================
echo Multi-Obstacle Biped Walking Test
echo ======================================================================
echo.
echo This script will test your biped robot with:
echo   - 5 obstacles
echo   - 10 obstacles  
echo   - 15 obstacles
echo   - 20 obstacles
echo.
echo Each configuration will run 100 episodes.
echo Estimated time: 20-40 minutes (depending on your system)
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo Starting tests...
echo.

python test_sac_multi_obstacles.py

echo.
echo ======================================================================
echo Tests completed!
echo Check the generated CSV files for detailed results.
echo ======================================================================
echo.
pause
