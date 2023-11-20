# Import the neccessary module
import gradio as gr
import pandas as pd
import os
from pandasai.llm import OpenAI
from pandasai import PandasAI
from pandasai.llm import OpenAI,Falcon,Starcoder
from pandasai import SmartDataframe,SmartDatalake

# Set default llm
llm=OpenAI(api_token=os.getenv(OPENAI_API_KEY))

def llm_function(llm_model):
  if llm_model=="OpenAi":
    llm=OpenAI(api_token=os.getenv(OPENAI_API_KEY))
  elif llm_model=="Falcon":
    llm=Falcon(api_token=os.getenv(HUGGINGFACE_API_KEY))
  else:
    llm=Starcoder(api_token=os.getenv(HUGGINGFACE_API_KEY))
  print(llm_model)

# Show the saved plot to the user
def show_plot():
  return "temp_chart.png"

# Upload the csv file
def upload_file(files):
    file_paths = [file.name for file in files]
    df=pd.read_csv(files.name)
    global model
    model=SmartDataframe(df,config={"llm":llm})
    return df.head(5)

# Generate the string response and plot according to the prompt
def response(prompt):
  global response
  response=model.chat(prompt)
  if response==None:
    return "Click on Show Plot Button"
  return response

# Gradio interface
with gr.Blocks() as demo:
  gr.Markdown("# CSV Analyzer")
  # Radio button to select the llm model
  list_llm=gr.Radio(["OpenAi","Falcon","Starcoder"],label="LLM Models")
  list_llm.change(llm_function,list_llm)

  # Button to upload the csv file
  upload_button = gr.UploadButton("Click to Browse CSV File", file_types=["CSV"])
  dataframe=gr.inputs.Dataframe()
  upload_button.upload(upload_file,upload_button, dataframe)

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
