from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os
import uuid
import pandas as pd
import re
from excel_actions import handle_excel_task

app = FastAPI()

# CORS setup to allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can later restrict to localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body schema (kept simple)
from pydantic import BaseModel
class ChatRequest(BaseModel):
    user_message: str
    excel_data: list

@app.get("/")
def home():
    return {"message": "Excel AI Assistant Backend Running ✅"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        message = req.user_message or ""
        data = req.excel_data

        # Convert incoming Excel JSON data to DataFrame
        if not data or len(data) < 1:
            return {"ok": False, "note": "❌ No Excel data provided."}
        df = pd.DataFrame(data[1:], columns=data[0])

        # ... earlier code unchanged ...

        # Create a version of the message that removes any brace-specified column names
        # so words inside { } do not influence intent detection.
        message_for_intent = re.sub(r"\{[^}]+\}", " ", message or "", flags=re.IGNORECASE).lower()
        # reduce multiple spaces
        message_for_intent = re.sub(r"\s+", " ", message_for_intent).strip()

        # quick helper to check whole-word presence to avoid partial matches
        def has_word(text, words):
            for w in words:
                # use word boundary to avoid matching substrings inside other words
                if re.search(rf"\b{re.escape(w)}\b", text):
                    return True
            return False

        # Use the cleaned message_for_intent for intent matching
        lm = message_for_intent

        # check "summary"/"statistics" before "sum"
        if has_word(lm, ["summary", "statistics", "describe", "overview"]):
            intent = "summary"
        elif any(phrase in lm for phrase in ["difference between max and min", "min and max", "max and min", "data spread", "range", "range of", "spread range"]):
            intent = "range"
        elif has_word(lm, ["chart", "plot", "bar", "scatter", "line", "boxplot", "hist", "pie", "heatmap", "area"]):
            intent = "chart"
        elif has_word(lm, ["regression", "predict"]):
            intent = "regression"
        elif has_word(lm, ["median", "middle value", "central value", "mid value"]):
            intent = "median"
        elif has_word(lm, ["mode", "most frequent", "most common", "frequent value"]):
            intent = "mode"
        elif has_word(lm, ["variance", "var", "spread", "variability", "how spread out"]):
            intent = "variance"
        elif has_word(lm, ["standard deviation", "std dev", "std", "deviation"]):
            intent = "stddev"
        elif has_word(lm, ["correlation", "relation", "relationship", "connection", "association", "compare", "linked with"]):
            intent = "correlation"
        elif has_word(lm, ["sum", "add", "add all"]):
            intent = "sum"
        elif has_word(lm, ["average", "mean", "avg"]):
            intent = "average"
        elif has_word(lm, ["minimum", "lowest", "smallest", "min value", "min"]):
            intent = "min"
        elif has_word(lm, ["maximum", "highest", "largest", "max value", "max"]):
            intent = "max"
        elif has_word(lm, ["count", "how many entries", "number of rows", "total rows"]):
            intent = "count"
        else:
            intent = "unknown"

        # Optional: debug logging (remove later)
        print(f"DEBUG intent detection -> message_for_intent: '{message_for_intent}'  intent: '{intent}'")

        # Execute corresponding logic
        result = handle_excel_task(intent, df, message)
        return result

    except Exception as e:
        return {"ok": False, "note": f"⚠️ Backend error: {str(e)}"}

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accept an uploaded Excel/CSV file, save it, parse sheets and return metadata + preview.
    Frontend will decide where/how to place this into the workbook.
    """
    try:
        # Save to disk with unique id
        file_ext = os.path.splitext(file.filename)[1].lower()
        file_id = f"{uuid.uuid4().hex}{file_ext}"
        dest_path = os.path.join(UPLOAD_DIR, file_id)
        with open(dest_path, "wb") as f:
            f.write(await file.read())

        # Try reading as Excel first, then CSV
        parsed = {}
        if file_ext in [".xls", ".xlsx", ".xlsm", ".xlsb"]:
            try:
                xls = pd.read_excel(dest_path, sheet_name=None, engine="openpyxl")
            except Exception:
                # fallback to any engine
                xls = pd.read_excel(dest_path, sheet_name=None)
            sheets = []
            for sheet_name, df in xls.items():
                # sanitize headers
                headers = [str(c) for c in df.columns.tolist()]
                rows = df.head(5).fillna("").values.tolist()
                sheets.append({"name": sheet_name, "headers": headers, "row_count": int(df.shape[0]), "preview": rows})
            parsed["type"] = "excel"
            parsed["sheets"] = sheets
        else:
            # try reading csv
            try:
                df = pd.read_csv(dest_path)
            except Exception:
                # try with different separator
                df = pd.read_csv(dest_path, sep=None, engine="python")
            headers = [str(c) for c in df.columns.tolist()]
            rows = df.head(5).fillna("").values.tolist()
            parsed["type"] = "csv"
            parsed["sheets"] = [{"name": "Sheet1", "headers": headers, "row_count": int(df.shape[0]), "preview": rows}]

        return JSONResponse({"ok": True, "file_id": file_id, "original_name": file.filename, "parsed": parsed})
    except Exception as e:
        return JSONResponse({"ok": False, "detail": f"Upload/parsing failed: {str(e)}"}, status_code=500)



@app.get("/uploaded_data/{file_id}")
def uploaded_data(file_id: str, sheet: int = 0):
    """
    Return the full uploaded sheet content as JSON:
    {
      "ok": True,
      "headers": [...],
      "rows": [...],  # includes header row optionally
      "rows_no_header": [...]  # rows only
    }
    """
    path = os.path.join(UPLOAD_DIR, file_id)
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "detail": "file not found"}, status_code=404)
    try:
        file_ext = os.path.splitext(path)[1].lower()
        if file_ext in [".xls", ".xlsx", ".xlsm", ".xlsb"]:
            # read specific sheet by index
            xls = pd.read_excel(path, sheet_name=None, engine="openpyxl")
            sheet_names = list(xls.keys())
            if sheet < 0 or sheet >= len(sheet_names):
                return JSONResponse({"ok": False, "detail": "sheet index out of range"}, status_code=400)
            df = xls[sheet_names[sheet]]
        else:
            df = pd.read_csv(path)
        headers = [str(c) for c in df.columns.tolist()]
        rows = df.fillna("").values.tolist()
        rows_no_header = df.fillna("").values.tolist()
        return JSONResponse({"ok": True, "headers": headers, "rows": rows, "rows_no_header": rows_no_header})
    except Exception as e:
        return JSONResponse({"ok": False, "detail": f"read failed: {str(e)}"}, status_code=500)

