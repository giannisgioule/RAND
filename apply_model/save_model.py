from tkinter import filedialog
from tkinter import *
import joblib

class ModelSave():

    def __init__(self,,model):
        self.model=model
    
    root=Tk()
    root.filename=filedialog.asksaveasfilename(\
    initialdir ="/",\
    title ="Select file",\
    defaultextension =".sav")

    joblib.dump(self.model,root.filename)
    
    return print("File saved")