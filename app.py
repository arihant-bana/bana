import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- Data Preparation ---
datasets = {
    "Wheat": "wheat_varieties_huge.csv",
    "Maize": "maize_varieties_comprehensive.csv",
    "Rice": "rice_varieties.csv",
}

prepped = {}

def prepare_crop(crop_name, filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    # Find columns
    state_col = next((c for c in df.columns if "state" in c), None)
    dist_col = next((c for c in df.columns if "district" in c), None)
    irrigation_col = next((c for c in df.columns if "irrigation" in c), None)
    sowing_col = next((c for c in df.columns if "sowing" in c or "season" in c), None)
    recom_col = next((c for c in df.columns if "recommend" in c), None)
    # Defensive check
    if None in [state_col, dist_col, irrigation_col, sowing_col, recom_col]:
        raise ValueError(f"Missing required columns for {crop_name}.")
    for col in [state_col, dist_col, irrigation_col, sowing_col]:
        df[col + "_clean"] = df[col].astype(str).str.strip().str.lower()
    df["primary_variety"] = df[recom_col].astype(str).apply(lambda x: x.split(";")[0].strip())
    le_state = LabelEncoder().fit(df[state_col].astype(str))
    le_dist = LabelEncoder().fit(df[dist_col].astype(str))
    le_irri = LabelEncoder().fit(df[irrigation_col].astype(str))
    le_sow = LabelEncoder().fit(df[sowing_col].astype(str))
    le_var = LabelEncoder().fit(df["primary_variety"].astype(str))
    df["state_enc"] = le_state.transform(df[state_col].astype(str))
    df["dist_enc"] = le_dist.transform(df[dist_col].astype(str))
    df["irri_enc"] = le_irri.transform(df[irrigation_col].astype(str))
    df["sow_enc"] = le_sow.transform(df[sowing_col].astype(str))
    df["var_enc"] = le_var.transform(df["primary_variety"].astype(str))
    features = ["state_enc", "dist_enc", "irri_enc", "sow_enc"]
    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(df[features], df["var_enc"])
    return {
        "df": df,
        "state_col": state_col,
        "dist_col": dist_col,
        "irrigation_col": irrigation_col,
        "sowing_col": sowing_col,
        "recom_col": recom_col,
        "le_state": le_state,
        "le_dist": le_dist,
        "le_irri": le_irri,
        "le_sow": le_sow,
        "le_var": le_var,
        "model": clf,
    }

for crop_name, path in datasets.items():
    prepped[crop_name] = prepare_crop(crop_name, path)

def safe_dropdown(choices):
    choices = [c for c in (choices or []) if pd.notna(c) and str(c).strip()]
    if not choices:
        return gr.update(choices=['No options'], value="No options", interactive=False)
    else:
        return gr.update(choices=choices, value=None, interactive=True)

def get_states(crop):
    if crop not in prepped:
        return []
    return sorted(prepped[crop]["le_state"].classes_)

def get_districts(crop, state):
    if not crop or not state or crop not in prepped:
        return []
    d = prepped[crop]
    df = d["df"]
    s_clean = state.strip().lower()
    districts_clean = df[df[d["state_col"] + "_clean"] == s_clean][d["dist_col"] + "_clean"].unique()
    display_names = []
    for d_clean in districts_clean:
        df_sub = df[df[d["dist_col"] + "_clean"] == d_clean]
        if not df_sub.empty:
            idx = df_sub.index[0]
            display_names.append(df_sub.at[idx, d["dist_col"]])
    return sorted(display_names)

def get_sowings(crop, state, district):
    if not crop or not state or not district or crop not in prepped:
        return []
    d = prepped[crop]
    df = d["df"]
    s_clean, d_clean = state.strip().lower(), district.strip().lower()
    rows = df[(df[d["state_col"] + "_clean"] == s_clean) & (df[d["dist_col"] + "_clean"] == d_clean)]
    return sorted(rows[d["sowing_col"]].unique())

def get_irrigations(crop, state, district, sowing):
    if not crop or not state or not district or not sowing or crop not in prepped:
        return []
    d = prepped[crop]
    df = d["df"]
    s_clean, d_clean, sow_clean = state.strip().lower(), district.strip().lower(), sowing.strip().lower()
    rows = df[(df[d["state_col"] + "_clean"] == s_clean) & 
              (df[d["dist_col"] + "_clean"] == d_clean) & 
              (df[d["sowing_col"] + "_clean"] == sow_clean)]
    return sorted(rows[d["irrigation_col"]].unique())

def recommend(crop, state, district, sowing, irrigation):
    if not all([crop, state, district, sowing, irrigation]):
        return "<div class='card-output'><b>⚠️ Please fill all selections.</b></div>"
    d = prepped[crop]
    try:
        X = pd.DataFrame([[
            d["le_state"].transform([state])[0], 
            d["le_dist"].transform([district])[0],
            d["le_irri"].transform([irrigation])[0],
            d["le_sow"].transform([sowing])[0]
        ]], columns=["state_enc","dist_enc","irri_enc","sow_enc"])
        pred_enc = d["model"].predict(X)[0]
        pred_var = d["le_var"].inverse_transform([pred_enc])[0]
        mask = (d["df"]["state_enc"] == X["state_enc"][0]) & (d["df"]["dist_enc"] == X["dist_enc"][0]) & \
               (d["df"]["irri_enc"] == X["irri_enc"][0]) & (d["df"]["sow_enc"] == X["sow_enc"][0])
        rec_varieties = "N/A"
        if mask.any():
            recs = d["df"].loc[mask, d["recom_col"]]
            if not recs.empty and pd.notna(recs.iloc[0]):
                rec_varieties = str(recs.iloc[0])
        # Modern css ensures strong contrast/visibility - use pure <pre> for output box, not monospace theme
        return f"""
        <div class='card-output'>
            <span class='reco-title'>{crop} Recommendation</span><br>
            <span class='reco-label'>ML Predicted Variety:</span>
            <span class='reco-value'>{pred_var}</span><br>
            <span class='reco-label'>Other recommended varieties:</span>
            <pre class='reco-pre'>{rec_varieties}</pre>
        </div>"""
    except Exception as e:
        return f"<div class='card-output'><b>⚠️ Prediction Error: {str(e)}</b></div>"

css = """
body, .gradio-container {
    background: linear-gradient(120deg,#a1c4fd,#c2e9fb 100%) !important;
    min-width: 1150px !important; max-width: 1350px !important;
    margin: 0 auto !important;
    font-family: "Inter",sans-serif;
    color: #23232b !important;
}
.logo-banner {
    display: flex; flex-direction: column; align-items: center; gap:.5em; margin-top:.7em;
    justify-content: center;
}
.logo-banner img {
    width: 90px; border-radius: 18px; box-shadow: 0 3px 20px #bcd6fc2c; display: block; margin:auto;
}
.card-output {
    background: #fff;
    border-radius: 19px;
    box-shadow: 0 6px 24px #1459b70d;
    padding: 2.1em 2em 1.3em 2em;
    margin-top: 1em;
    min-width: 440px;
    max-width: 660px;
    margin-left: auto; margin-right: auto;
    color: #1b1a1d;
}
.reco-title {
    color: #2761b0; font-weight: bold; font-size:1.13em; margin-bottom: 3px;
}
.reco-label {
    color: #285f54; font-weight: 600; font-size: 1.09em;
}
.reco-value {
    font-weight: 710; font-size: 1.37em; color: #264796; letter-spacing: .2px;
    display:block; margin-bottom:.45em;
}
.reco-pre {
    background: #f4f8fe; color: #28363f; border-radius:6px; margin-top:7px; margin-bottom:3px; padding:6px 13px 6px 6px; font-size:1.09em;
    font-family: 'Segoe UI', Arial, Helvetica, sans-serif; font-weight:500;
    min-width:260px; word-break:break-all; white-space:pre-line; border:1px solid #b7d4f2;
}
.gr-radio input[type=radio] {
    border-radius: 50% !important; width:19px; height:19px; accent-color:#3777d6;
    box-shadow:0 1px 5px #3777d635; margin-right:6px;
}
.gr-radio label {
    color: #1d3355; font-size:1.09em; font-weight:540; padding-left:4px;
}
.gr-dropdown select, .gr-dropdown * {
    font-size:1.09em; background:#f8fafd; border:1px solid #b6d3f2;
    border-radius:7px; color:#222;
}
.gr-button {
    background: linear-gradient(90deg,#3777d5 0%,#78caff 100%)!important;
    border-radius:15px; color:#fff;font-weight:700;
    font-size:1.11em; box-shadow:0 2px 12px #3777d545;
    padding:10px 24px; margin-top:10px;
}
.gr-row { flex-wrap:nowrap !important;}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("""
        <div class='logo-banner'>
            <img src='https://iili.io/FOfoIqX.md.png' alt='Logo'>
            <h1 style='font-size:2.16em;color:#3c599c;margin:6px 0;text-align:center'>Crop Variety Recommender</h1>
            <p style='color:#457aae;font-size:1.07em;text-align:center'>• Wheat • Maize • Rice</p>
            <p style='color:#457aae;font-size:1.07em;text-align:center'>Made By <strong>Arihant Bana</strong> and Kanav Grover of class XI B</p>
        </div>
    """)
    crop_radio = gr.Radio(label="Crop", choices=list(datasets.keys()), value= None, interactive=True)
    state_dd = gr.Dropdown(label="State", interactive=True, choices=[])
    district_dd = gr.Dropdown(label="District", interactive=True, choices=[])
    sowing_dd = gr.Dropdown(label="Sowing Timing/Season", interactive=True, choices=[])
    irrigation_dd = gr.Dropdown(label="Irrigation Status", interactive=True, choices=[])
    recommend_btn = gr.Button("Get Recommendation")
    rec_output = gr.HTML("")

    crop_radio.change(lambda c: [
        safe_dropdown(get_states(c)),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None)
    ], crop_radio, [state_dd, district_dd, sowing_dd, irrigation_dd])

    state_dd.change(lambda c, s: [
        safe_dropdown(get_districts(c, s)),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None)
    ], [crop_radio, state_dd], [district_dd, sowing_dd, irrigation_dd])

    district_dd.change(lambda c, s, d: [
        safe_dropdown(get_sowings(c, s, d)),
        gr.update(choices=[], value=None)
    ], [crop_radio, state_dd, district_dd], [sowing_dd, irrigation_dd])

    sowing_dd.change(lambda c, s, d, sw: safe_dropdown(get_irrigations(c, s, d, sw)),
                     [crop_radio, state_dd, district_dd, sowing_dd], irrigation_dd)

    recommend_btn.click(recommend,
                       [crop_radio, state_dd, district_dd, sowing_dd, irrigation_dd],
                       rec_output)

    gr.Markdown("""
        <div style='margin-top:2em; text-align:center; color:#91b2e3; font-size:1.03em'>
            © 2025 <strong>Arihant Bana</strong>
        </div>
    """)

demo.launch()
