Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "pythonw """ & WScript.ScriptFullName & "\..\predictive_maintenance_gui.py""", 0
Set WshShell = Nothing