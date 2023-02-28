import tkinter as tk


def calculate():
    # perform calculation and update result label
    pass


def clear():
    # clear input and result label
    pass


def add_digit(digit):
    # add digit to input label
    pass


def add_operator(operator):
    # add operator to input label
    pass


# create GUI window
window = tk.Tk()
window.title("Calculator")

# create input and result labels
input_label = tk.Label(window, text="")
input_label.pack()

result_label = tk.Label(window, text="")
result_label.pack()

# create digit buttons
button_1 = tk.Button(window, text="1", command=lambda: add_digit(1))
button_1.pack()

button_2 = tk.Button(window, text="2", command=lambda: add_digit(2))
button_2.pack()

# create operator buttons
button_plus = tk.Button(window, text="+", command=lambda: add_operator("+"))
button_plus.pack()

button_minus = tk.Button(window, text="-", command=lambda: add_operator("-"))
button_minus.pack()

# create calculation and clear buttons
button_calculate = tk.Button(window, text="=", command=calculate)
button_calculate.pack()

button_clear = tk.Button(window, text="C", command=clear)
button_clear.pack()

# run GUI
window.mainloop()
