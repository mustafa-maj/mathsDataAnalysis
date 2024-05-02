import tkinter as tk

# Create the main window
window = tk.Tk()
window.title("Canvas Example")

# Create the canvas
canvas = tk.Canvas(window, width=200, height=100)
canvas.pack()

# Draw a line on the canvas
new_line = canvas.create_line(0, 0, 200, 100)

# Run the main loop
window.mainloop()
