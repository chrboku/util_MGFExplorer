import ast

ast.parse(open("src/msmsexplorer/app.py", encoding="utf-8").read())
ast.parse(open("src/msmsexplorer/gui_components.py", encoding="utf-8").read())
print("OK")
