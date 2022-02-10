from tkinter import *
from PIL import ImageGrab, Image
import PIL.ImageOps    
import cv2
import numpy as np
import torch
from src.shabaka_net import ShabakaNet
from src.utils import predict, img_to_tensor

current_x, current_y = 0,0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ShabakaNet()
model.load_state_dict(torch.load("model/shabakanet.pt"))

def locate_xy(event):
    
    global current_x, current_y

    current_x, current_y = event.x, event.y

def addLine(event):
    
    global current_x, current_y
    
    canvas.create_line((current_x,current_y,event.x,event.y),fill = 'black', width = 6)
    current_x, current_y = event.x, event.y

def new_canvas():
    
    canvas.delete('all')
    output.delete(0, END)
    #display_pallete()
    
def predict_drawing():
    x0 = Canvas.winfo_rootx(canvas)
    y0 = Canvas.winfo_rooty(canvas)
    x1 = x0 + Canvas.winfo_width(canvas)
    y1 = y0 + Canvas.winfo_height(canvas)
    img = ImageGrab.grab((x0, y0, x1, y1))
    tens = img_to_tensor(img)
    conf, letter = predict(model, device, tens)
    output.delete(0, END)
    output.insert(0, "%s (%2.2f%%)"%(letter, conf))

window = Tk()

window.title('Where is my arabic letter')
window.geometry('600x600')
window.tk.call('wm', 'iconphoto', window._w, PhotoImage(file='icon.png'))

window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)

menubar = Menu(window)
window.config(menu = menubar)
submenu = Menu(menubar,tearoff=0)

menubar.add_cascade(label='File', menu=submenu)
submenu.add_command(label='New Board', command=new_canvas)

canvas= Canvas(window,width=128,height=128,background='white') 
canvas.grid(row=0,column=0,sticky='nsew')#rajoutercolumnspan=2 pour quil prenne 2 col

canvas.bind('<Button-1>', locate_xy)
canvas.bind('<B1-Motion>',addLine)

prediction_btn = Button(window,text="prediction",padx=300,font='Arial',command=predict_drawing)
prediction_btn.grid(row=1, column=0,sticky=W,columnspan=2)
output = Entry(window, font = "Arial 44", justify="center", bg='grey',width=100,borderwidth=10)
output.grid(row=2, column=0, columnspan=2)

window.mainloop()
#retrecir le white board (canva)pour que je puisse compresser limage facilement