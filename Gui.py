import tkinter as tk
from tkinter import ttk
from webScrapper import Scrapper

web = Scrapper()
class GUI():
    global emote
    def display_message(self,url):
        web.launchBrowser(url)

    def guiFunction(self,message,url):
        # Create the main window
        window = tk.Tk()
        window.title("Electronic Style GUI")

        # Set a custom font for the label
        custom_font = ("Digital-7", 30)

        # Create a themed style
        style = ttk.Style()
        style.configure("TLabel", font=custom_font)

        # Create a label with a 3D and electronic style font
        label = ttk.Label(window, text=message, style='TLabel')
        label.pack(pady=20)

        # Create a button with a modern style
        button = ttk.Button(window, text="Links", command=self.display_message(url))
        button.pack(pady=10)

        # Start the main loop
        window.mainloop()
