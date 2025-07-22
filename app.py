import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import openai
import os

# ------------------ ENVIRONMENT VARIABLE (OpenAI API Key) ------------------
openai.api_key = "sk-proj-your_key_here"  # <== Paste your OpenAI key here

# ------------------ DATA LOADING ------------------
CSVFILE = "wheat_varieties_huge.csv"
df = pd.read_csv(CSVFILE)
df.columns = df.columns.str.strip()

cols_to_strip = ['State', 'District', 'Sowing_Timing', 'Irrigation_Status', 'Recommended_Varieties']
df[cols_to_strip] = df[cols_to_strip].apply(lambda x: x.str.strip())

# Label encode
le_state = LabelEncoder()
le_district = LabelEncoder()
le_sowing = LabelEncoder()
le_irrigation = LabelEncoder()
le_recommend = LabelEncoder()

df["State_enc"] = le_state.fit_transform(df["State"])
df["District_enc"] = le_district.fit_transform(df["District"])
df["Sowing_Timing_enc"] = le_sowing.fit_transform(df["Sowing_Timing"])
df["Irrigation_Status_enc"] = le_irrigation.fit_transform(df["Irrigation_Status"])
df["Recommended_Varieties_enc"] = le_recommend.fit_transform(df["Recommended_Varieties"])

X = df[["State_enc", "District_enc", "Sowing_Timing_enc", "Irrigation_Status_enc"]]
y = df["Recommended_Varieties_enc"]
model = RandomForestClassifier()
model.fit(X, y)

def recommend_variety(state, district, sowing_time, irrigation_status):
    try:
        input_data = pd.DataFrame([[
            le_state.transform([state])[0],
            le_district.transform([district])[0],
            le_sowing.transform([sowing_time])[0],
            le_irrigation.transform([irrigation_status])[0]
        ]], columns=["State_enc", "District_enc", "Sowing_Timing_enc", "Irrigation_Status_enc"])

        pred_enc = model.predict(input_data)[0]
        pred_variety = le_recommend.inverse_transform([pred_enc])[0]
        return pred_variety
    except Exception as e:
        return f"Error: {str(e)}"

def update_districts(state):
    filtered = df[df["State"] == state]
    districts = sorted(filtered["District"].unique())
    return gr.update(choices=districts)

# ------------------ OpenAI Chat Function ------------------
def ask_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------ CUSTOM CSS ------------------
css = """
#chat-wrapper {
    border: 2px solid #4a90e2;
    padding: 12px;
    border-radius: 12px;
    background: #f7fbff;
    margin-top: 20px;
}
#askbtn {
    background-color: #4a90e2;
    color: white;
    padding: 10px 16px;
    border-radius: 8px;
    margin-top: 10px;
}
#askbtn:hover {
    background-color: #357ab7;
}
#chat-response {
    min-height: 100px;
    padding: 10px;
    background: #eef4fd;
    border: 1px solid #c9e0ff;
    border-radius: 8px;
    font-family: 'Segoe UI', sans-serif;
}
"""

# ------------------ UI ------------------
with gr.Blocks(css=css) as demo:
    gr.Markdown("<h1 style='text-align:center;'>🌾 AI Crop Variety Recommender + ChatGPT</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            state_input = gr.Dropdown(label="State", choices=sorted(df["State"].unique()))
            district_input = gr.Dropdown(label="District", choices=[])
            sowing_input = gr.Dropdown(label="Sowing Time", choices=sorted(df["Sowing_Timing"].unique()))
            irrigation_input = gr.Dropdown(label="Irrigation Status", choices=sorted(df["Irrigation_Status"].unique()))
            submit_btn = gr.Button("Get Recommended Variety")
        with gr.Column(scale=1):
            output = gr.Textbox(label="Recommended Variety")

    state_input.change(fn=update_districts, inputs=state_input, outputs=district_input)
    submit_btn.click(fn=recommend_variety, inputs=[state_input, district_input, sowing_input, irrigation_input], outputs=output)

    # ------------------ ChatGPT Section ------------------
    with gr.Column(elem_id="chat-wrapper"):
        gr.Markdown("<h3 style='text-align:center;color:#234;margin-bottom:10px'>💬 Ask OpenAI Anything</h3>")
        user_prompt = gr.Textbox(label="Your Question", lines=3, placeholder="Type your question...")
        ask_button = gr.Button("Ask OpenAI", elem_id="askbtn")
        openai_output = gr.Textbox(label="GPT Response", elem_id="chat-response", lines=6)

        def show_loading(_): return "⏳ Thinking..."
        ask_button.click(fn=show_loading, inputs=user_prompt, outputs=openai_output)
        ask_button.click(fn=ask_openai, inputs=user_prompt, outputs=openai_output)

demo.launch()
