# Import the neccessary module
import gradio as gr
import pandas as pd
import os
from pandasai.llm import OpenAI
from pandasai import PandasAI
from pandasai.llm import OpenAI,Falcon,Starcoder
from pandasai import SmartDataframe,SmartDatalake


# Show the saved plot to the user
def show_plot():
  return "temp_chart.png"

# Upload the csv file
def upload_file(files,llm_model,api_key):
    file_paths = [file.name for file in files]
    df=pd.read_csv(files.name)
    if llm_model=="OpenAi":
      llm=OpenAI(api_token=api_key)
    elif llm_model=="Falcon":
      llm=Falcon(api_token=api_key)
    else:
      llm=Starcoder(api_token=api_key)
    global model
    model=SmartDataframe(df,config={"llm":llm})
    return df.head(5)

# Generate the string response and plot according to the prompt
def response(prompt):
  response=model.chat(prompt)
  if response==None:
    return "Click on Show Plot Button"
  return response

# Gradio interface
with gr.Blocks() as demo:
  # Radio button to select the llm model
  with gr.Row():
    with gr.Column():
      list_llm=gr.Radio(["OpenAi","Falcon","Starcoder"],label="LLM Models",default="OpenAi")
    with gr.Column():
      api_key=gr.Textbox(label="LLM Model Api Key")

  # Button to upload the csv file
  upload_button = gr.UploadButton("Click to Browse CSV File", file_types=["CSV"])
  dataframe=gr.inputs.Dataframe()
  upload_button.upload(upload_file, [upload_button,list_llm,api_key], dataframe)

  # Input output box for prompt and string response
  with gr.Row():
    with gr.Column():
      prompt=gr.inputs.Textbox(label="Prompt")
    with gr.Column():
      output=gr.Textbox(label="Output")
  submit=gr.Button("Generate Response")
  submit.click(response,prompt,output)

  plot=gr.Image(shape=(50,50))
  plot_button=gr.Button("Show Plot")
  plot_button.click(show_plot,None,plot)



# Launch the interface with public link with  authentication
demo.launch(debug=True)
