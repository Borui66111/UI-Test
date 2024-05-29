pip install -r requirements.txt
# pyinstaller --onefile --add-data "llm.py:." --add-data "ppt.py:." --additional-hooks-dir=. --hidden-import="pkg_resources.py2_warn" --requirements requirements.txt test.py
