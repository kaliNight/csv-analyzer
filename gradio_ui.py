# Import the neccessary module
import gradio as gr
import pandas as pd
from pandasai.llm import OpenAI
from pandasai import PandasAI
from pandasai.llm import OpenAI,Falcon,Starcoder
from pandasai import SmartDataframe,SmartDatalake

# Made the OpenAi as default llm
llm=OpenAI(api_token="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")

# Ask the user for selecting the llm model from radio button
def llm_function(llm_model):
  if llm_model=="OpenAi":
    llm=OpenAI(api_token="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")
  elif llm_model=="Falcon":
    llm=Falcon(api_token="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")
  else:
    llm=Starcoder(api_token="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")

# Upload the csv file
def upload_file(files):
    file_paths = [file.name for file in files]
    df=pd.read_csv(files.name)
    global model
    model=SmartDataframe(df,config={"llm":llm})
    return df.head(5)

# Generate the string response and plot according to the prompt
# Saving the image of plot in colab for user download purpose
def response(prompt):
  response=model.chat(prompt)
  if type(response)!=str:
    return "The plot image(temp_chart.png) is saved in colab you can download it from there"
  return response

# Gradio interface
with gr.Blocks() as demo:
  # Radio button to select the llm model
    list_llm=gr.Radio(["OpenAi","Falcon","Starcoder"],label="LLM Models",default="OpenAi")
    list_llm.change(llm_function,list_llm)

  # Button to upload the csv file
    upload_button = gr.UploadButton("Click to Browse CSV File", file_types=["CSV"])
    dataframe=gr.inputs.Dataframe()
    upload_button.upload(upload_file, upload_button, dataframe)

  # Input output box for prompt and string response
    with gr.Row():
      with gr.Column():
        prompt=gr.inputs.Textbox(label="Prompt")
      with gr.Column():
        output=gr.Textbox(label="Output")
    submit=gr.Button("Generate Response")
    submit.click(response,prompt,output)


# Launch the interface with public link with  authentication