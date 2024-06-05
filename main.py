import math
import sympy as sp
import numpy as np
import tkinter as tk
import smtplib
import webbrowser
from tkinter import simpledialog, messagebox
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown
from kivy.uix.boxlayout import BoxLayout
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# Function to solve linear equations
def solve_linear_equation(equation):
    symbols = set(ch for ch in equation if ch.isalpha())  # Get all alphabets from the equation
    symbols = [sp.symbols(symbol) for symbol in symbols]  # Create symbols dynamically
    lhs, rhs = equation.split('=')
    for symbol in symbols:
        lhs = lhs.replace(symbol.name, f'*{symbol.name}')  # Replace each symbol in lhs with '*symbol'
    lhs = sp.sympify(lhs)
    rhs = sp.sympify(rhs)
    solution = sp.solve(lhs - rhs, symbols)
    return solution


# Function to solve quadratic equations
def solve_quadratic_equation(a, b, c):
    x = sp.symbols('x')
    solutions = sp.solve(a * x ** 2 + b * x + c, x)
    return solutions


# Function to solve polynomial equations
def solve_polynomial_equation(coefficients):
    x = sp.symbols('x')
    polynomial = sum(coef * x ** i for i, coef in enumerate(reversed(coefficients)))
    roots = sp.solve(polynomial, x)
    return roots


# Function to solve cubic equations
def solve_cubic_equation(a, b, c, d):
    x = sp.symbols('x')
    cubic_eq = a * x ** 3 + b * x ** 2 + c * x + d
    roots = sp.solve(cubic_eq, x)
    return roots


# Function to differentiate expressions
def differentiate(expression):
    x = sp.symbols('x')
    expr = sp.sympify(expression)
    derivative = sp.diff(expr, x)
    return derivative


# Function to integrate expressions
def integrate(expression):
    x = sp.symbols('x')
    expr = sp.sympify(expression)
    integral = sp.integrate(expr, x)
    return integral


# Function to invert a matrix
def matrix_inversion(matrix):
    try:
        inv_matrix = np.linalg.inv(matrix)
        return inv_matrix
    except np.linalg.LinAlgError:
        return None


# Function to calculate determinant of a matrix
def matrix_determinant(matrix):
    try:
        det = np.linalg.det(matrix)
        return det
    except np.linalg.LinAlgError:
        return None


from kivy.uix.button import Button

class DestiSolverApp(App):
    def build(self):
        layout = GridLayout(cols=2, padding=10)

        # Add welcome message
        welcome_message = "Welcome to DestiSolver AI"
        self.welcome_label = Label(text=welcome_message, font_size=24)
        layout.add_widget(self.welcome_label)

        # Add logo image
        try:
            logo = "C:/Users/ENGR. DESTINY/Downloads/Background/Backgroud for Ai.jpg"
            self.logo_label = Label(text="DestiSolver AI")
        except FileNotFoundError:
            self.logo_label = Label(text="DestiSolver AI")
        layout.add_widget(self.logo_label)

        self.label = Label(text="Select an option:")
        layout.add_widget(self.label)

        options = [
            "Select Your Problem",
            "Simple Calculator Add, Sub, Divide, Multiply",
            "Solve linear equation",
            "Solve quadratic equation",
            "Solve polynomial equation",
            "Solve cubic equation",
            "Differentiate expression",
            "Integrate expression",
            "Exponential functions",
            "Trigonometric functions",
            "Matrix Inversion",
            "Matrix Determinant",
            "None Of The Above"
        ]
        self.option_menu = DropDown()
        for option in options:
            btn = Button(text=option, size_hint_y=None, height=40)
            btn.bind(on_release=lambda btn: self.option_selected(btn.text))
            self.option_menu.add_widget(btn)

        self.option_button = Button(text='Select Your Problem', size_hint=(None, None), width=200, height=40)
        self.option_button.bind(on_release=self.option_menu.open)
        layout.add_widget(self.option_button)

        self.suggest_button = Button(text="Suggest Questions", size_hint=(None, None), width=200, height=40)
        self.suggest_button.bind(on_release=self.suggest_questions)
        layout.add_widget(self.suggest_button)

        self.run_button = Button(text="Run", size_hint=(None, None), width=200, height=40)
        self.run_button.bind(on_release=self.run_option)
        layout.add_widget(self.run_button)

        self.about_button = Button(text="About", size_hint=(None, None), width=200, height=40)
        self.about_button.bind(on_release=self.display_about)
        layout.add_widget(self.about_button)

        self.instruction_button = Button(text="Instructions", size_hint=(None, None), width=200, height=40)
        self.instruction_button.bind(on_release=self.display_instructions)
        layout.add_widget(self.instruction_button)

        self.social_media_button = Button(text="Follow us on Social Media", size_hint=(None, None), width=200, height=40)
        self.social_media_button.bind(on_release=self.redirect_to_social_media)
        layout.add_widget(self.social_media_button)

        self.exit_button = Button(text="Exit", size_hint=(None, None), width=200, height=40)
        self.exit_button.bind(on_release=self.exit_app)
        layout.add_widget(self.exit_button)

        self.current_option = None

        return layout

    def display_instructions(self, instance):
        instructions_text = (
            "Instructions:\n\n"
            "1. Select your problem from the dropdown menu.\n"
            "2. Read the problem carefully and input the required information.\n"
            "3. Click on 'Suggest Questions' to get example questions related to the selected problem.\n"
            "4. Click on 'Run' to solve the problem.\n"
            "5. You can also click on 'About' to know more about DestiSolver AI.\n"
            "6. To exit the application, click on 'Exit'.\n\n"
            "Additional Features:\n\n"
            "- Click on 'Instructions' to view these instructions again.\n"
            "- Click on 'Follow us on Social Media' to connect with us on various platforms."
        )
        popup = Popup(title='Instructions', content=Label(text=instructions_text), size_hint=(None, None),
                      size=(400, 400))
        back_button = Button(text="Back to Main Menu", size_hint=(None, None), width=200, height=40)
        back_button.bind(on_release=popup.dismiss)
        popup.content.add_widget(back_button)
        popup.open()

    # Add the redirect_to_social_media method
    def redirect_to_social_media(self, instance):
        # Open the social media profiles in a web browser or integrate with APIs if available
        # Example:
        webbrowser.open_new_tab('https://www.facebook.com/Destisolverai')
        webbrowser.open_new_tab('https://twitter.com/Destisolverai')
        webbrowser.open_new_tab('https://www.instagram.com/Destisolverai')
        webbrowser.open_new_tab('https://www.linkedin.com/Destisolverai')

    def option_selected(self, text):
        self.option_button.text = text
        self.option_menu.dismiss()
        self.current_option = text

    def suggest_questions(self, instance):
        selected_option = self.option_button.text
        suggestions = {
            "Simple Calculator Add, Sub, Divide, Multiply": ["2 + 3", "4 * 5", "6 / 2", "7 - 4"],
            "Solve linear equation": ["2x + 3 = 0", "5x - 10 = 0"],
            "Solve quadratic equation": ["1x^2 - 3x + 2", "2x^2 + 4x + 2"],
            "Solve polynomial equation": ["3x^3 - 2x^2 + x - 5", "x^4 - 16"],
            "Solve cubic equation": ["x^3 + 2x^2 + 3x + 4", "x^3 - 6x^2 + 11x - 6"],
            "Differentiate expression": ["x^2 + 2x + 1", "sin(x) * cos(x)"],
            "Integrate expression": ["x^2", "1 / x"],
            "Exponential functions": ["e^x", "2^x"],
            "Trigonometric functions": ["sin(x)", "cos(x)", "tan(x)"],
            "Matrix Inversion": ["[[1, 2], [3, 4]]"],
            "Matrix Determinant": ["[[1, 2], [3, 4]]"]
        }
        suggestion_text = '\n'.join(suggestions.get(selected_option, ["No suggestions available."]))
        popup = Popup(title='Suggestions',
                      content=Label(text=f"Suggestions for {selected_option}:\n\n{suggestion_text}"),
                      size_hint=(None, None), size=(400, 400))
        back_button = Button(text="Back to Main Menu", size_hint=(None, None), width=200, height=40)
        back_button.bind(on_release=popup.dismiss)
        popup.content.add_widget(back_button)
        popup.open()
        popup.open()

    def run_option(self, instance):
        if self.current_option:
            self.show_instructions()

        popup = Popup(title='Instructions',
                      content=Label(text=instructions),
                      size_hint=(None, None), size=(400, 400))

        # Add next button to enter the solver
        next_button = Button(text="Next", size_hint=(None, None), size=(200, 40))
        next_button.bind(on_release=self.enter_solver)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(popup.content)
        layout.add_widget(next_button)

        popup.content = layout
        popup.open()

    def enter_solver(self, instance):
        self.process_option()

    def run_option(self, instance):
        self.process_option()

    def process_option(self):
        if self.current_option:
            if self.current_option == "Simple Calculator Add, Sub, Divide, Multiply":
                self.calculate()
            elif self.current_option == "Solve linear equation":
                self.solve_linear()
            elif self.current_option == "Solve quadratic equation":
                self.solve_quadratic()
            elif self.current_option == "Solve polynomial equation":
                self.solve_polynomial()
            elif self.current_option == "Solve cubic equation":
                self.solve_cubic()
            elif self.current_option == "Differentiate expression":
                self.differentiate_expression()
            elif self.current_option == "Integrate expression":
                self.integrate_expression()
            elif self.current_option == "Exponential functions":
                self.exponential_functions()
            elif self.current_option == "Trigonometric functions":
                self.trigonometric_functions()
            elif self.current_option == "Matrix Inversion":
                self.matrix_inversion()
            elif self.current_option == "Matrix Determinant":
                self.matrix_determinant()
            elif self.current_option == "None Of The Above":
                self.general_answer()

    def calculate(self):
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(text="Enter values to calculate (Add: +, Subtract: -, Multiply: *, Divide: /):"))
        expression_input = TextInput()
        content.add_widget(expression_input)

        popup = Popup(title='Calculate', content=content, size_hint=(None, None), size=(600, 400), auto_dismiss=False)

        def on_calculate(instance):
            expression = expression_input.text
            if expression:
                try:
                    result = eval(expression, {'__builtins__': None},
                                  {'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                                   'exp': math.exp,
                                   'log': math.log, 'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh})
                    result_label.text = f"The Answer Is =: {result}"
                except Exception as e:
                    result_label.text = f"Error: {str(e)}"

        result_label = Label(text="")
        calculate_button = Button(text="Calculate")
        calculate_button.bind(on_release=on_calculate)
        content.add_widget(result_label)
        content.add_widget(calculate_button)

        # Add back button
        back_button = Button(text="Back to Main Menu", size_hint=(None, None), size=(200, 40))
        back_button.bind(on_release=lambda btn: popup.dismiss())
        content.add_widget(back_button)

        popup.open()

    def solve_linear(self):
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(text="Enter linear equation (e.g., 2x + 3 = 0):"))
        equation_input = TextInput()
        content.add_widget(equation_input)

        popup = Popup(title='Solve Linear Equation', content=content, size_hint=(None, None), size=(600, 400),
                      auto_dismiss=False)

        def on_solve(instance):
            equation = equation_input.text
            if equation:
                try:
                    solution = solve_linear_equation(equation)
                    result_label.text = f"The solution is: {solution}"
                except Exception as e:
                    result_label.text = f"Error: {str(e)}"

        result_label = Label(text="")
        solve_button = Button(text="Solve")
        solve_button.bind(on_release=on_solve)
        content.add_widget(result_label)
        content.add_widget(solve_button)

        # Add back button
        back_button = Button(text="Back to Main Menu", size_hint=(None, None), size=(200, 40))
        back_button.bind(on_release=lambda btn: popup.dismiss())
        content.add_widget(back_button)

        popup.open()

    def solve_quadratic(self):
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(text="Enter coefficients a, b, c for ax^2 + bx + c = 0:"))
        a_input = TextInput(hint_text="a")
        b_input = TextInput(hint_text="b")
        c_input = TextInput(hint_text="c")
        content.add_widget(a_input)
        content.add_widget(b_input)
        content.add_widget(c_input)

        popup = Popup(title='Solve Quadratic Equation', content=content, size_hint=(None, None), size=(600, 400),
                      auto_dismiss=False)

        def on_solve(instance):
            try:
                a = float(a_input.text)
                b = float(b_input.text)
                c = float(c_input.text)
                solutions = solve_quadratic_equation(a, b, c)
                result_label.text = f"The solutions are: {solutions}"
            except Exception as e:
                result_label.text = f"Error: {str(e)}"

        result_label = Label(text="")
        solve_button = Button(text="Solve")
        solve_button.bind(on_release=on_solve)
        content.add_widget(result_label)
        content.add_widget(solve_button)

        # Add back button
        back_button = Button(text="Back to Main Menu", size_hint=(None, None), size=(200, 40))
        back_button.bind(on_release=lambda btn: popup.dismiss())
        content.add_widget(back_button)

        popup.open()

    def solve_polynomial(self):
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(
            text="Enter polynomial coefficients as comma-separated values (e.g., 1, -3, 2 for x^2 - 3x + 2):"))
        coefficients_input = TextInput()
        content.add_widget(coefficients_input)

        popup = Popup(title='Solve Polynomial Equation', content=content, size_hint=(None, None), size=(600, 400),
                      auto_dismiss=False)

        def on_solve(instance):
            coefficients = coefficients_input.text.split(',')
            coefficients = [float(coef) for coef in coefficients]
            if coefficients:
                try:
                    solutions = solve_polynomial_equation(coefficients)
                    result_label.text = f"The solutions are: {solutions}"
                except Exception as e:
                    result_label.text = f"Error: {str(e)}"

        result_label = Label(text="")
        solve_button = Button(text="Solve")
        solve_button.bind(on_release=on_solve)
        content.add_widget(result_label)
        content.add_widget(solve_button)

        # Add back button
        back_button = Button(text="Back to Main Menu", size_hint=(None, None), size=(200, 40))
        back_button.bind(on_release=lambda btn: popup.dismiss())
        content.add_widget(back_button)

        popup.open()

    def solve_cubic(self):
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(text="Enter coefficients a, b, c, d for ax^3 + bx^2 + cx + d = 0:"))
        a_input = TextInput(hint_text="a")
        b_input = TextInput(hint_text="b")
        c_input = TextInput(hint_text="c")
        d_input = TextInput(hint_text="d")
        content.add_widget(a_input)
        content.add_widget(b_input)
        content.add_widget(c_input)
        content.add_widget(d_input)

        popup = Popup(title='Solve Cubic Equation', content=content, size_hint=(None, None), size=(600, 400),
                      auto_dismiss=False)

        def on_solve(instance):
            try:
                a = float(a_input.text)
                b = float(b_input.text)
                c = float(c_input.text)
                d = float(d_input.text)
                solutions = solve_cubic_equation(a, b, c, d)
                result_label.text = f"The solutions are: {solutions}"
            except Exception as e:
                result_label.text = f"Error: {str(e)}"

        result_label = Label(text="")
        solve_button = Button(text="Solve")
        solve_button.bind(on_release=on_solve)
        content.add_widget(result_label)
        content.add_widget(solve_button)

        # Add back button
        back_button = Button(text="Back to Main Menu", size_hint=(None, None), size=(200, 40))
        back_button.bind(on_release=lambda btn: popup.dismiss())
        content.add_widget(back_button)

        popup.open()

    def differentiate_expression(self):
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(text="Enter the expression to differentiate (e.g., x^2 + 2x + 1):"))
        expression_input = TextInput()
        content.add_widget(expression_input)

        popup = Popup(title='Differentiate Expression', content=content, size_hint=(None, None), size=(600, 400),
                      auto_dismiss=False)

        def on_differentiate(instance):
            expression = expression_input.text
            if expression:
                try:
                    result = differentiate(expression)
                    result_label.text = f"The derivative is: {result}"
                except Exception as e:
                    result_label.text = f"Error: {str(e)}"

        result_label = Label(text="")
        differentiate_button = Button(text="Differentiate")
        differentiate_button.bind(on_release=on_differentiate)
        content.add_widget(result_label)
        content.add_widget(differentiate_button)

        # Add back button
        back_button = Button(text="Back to Main Menu", size_hint=(None, None), size=(200, 40))
        back_button.bind(on_release=lambda btn: popup.dismiss())
        content.add_widget(back_button)

        popup.open()

    def integrate_expression(self):
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(text="Enter the expression to integrate (e.g., x^2):"))
        expression_input = TextInput()
        content.add_widget(expression_input)

        popup = Popup(title='Integrate Expression', content=content, size_hint=(None, None), size=(600, 400),
                      auto_dismiss=False)

        def on_integrate(instance):
            expression = expression_input.text
            if expression:
                try:
                    result = integrate(expression)
                    result_label.text = f"The integral is: {result}"
                except Exception as e:
                    result_label.text = f"Error: {str(e)}"

        result_label = Label(text="")
        integrate_button = Button(text="Integrate")
        integrate_button.bind(on_release=on_integrate)
        content.add_widget(result_label)
        content.add_widget(integrate_button)

        # Add back button
        back_button = Button(text="Back to Main Menu", size_hint=(None, None), size=(200, 40))
        back_button.bind(on_release=lambda btn: popup.dismiss())
        content.add_widget(back_button)

        popup.open()

    def exponential_functions(self):
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(text="Enter the base and exponent (e.g., 2^3):"))
        expression_input = TextInput()
        content.add_widget(expression_input)

        popup = Popup(title='Exponential Functions', content=content, size_hint=(None, None), size=(600, 400),
                      auto_dismiss=False)

        def on_calculate(instance):
            expression = expression_input.text
            if expression:
                try:
                    base, exponent = expression.split('^')
                    base = float(base)
                    exponent = float(exponent)
                    result = base ** exponent
                    result_label.text = f"The result is: {result}"
                except Exception as e:
                    result_label.text = f"Error: {str(e)}"

        result_label = Label(text="")
        calculate_button = Button(text="Calculate")
        calculate_button.bind(on_release=on_calculate)
        content.add_widget(result_label)
        content.add_widget(calculate_button)

        # Add back button
        back_button = Button(text="Back to Main Menu", size_hint=(None, None), size=(200, 40))
        back_button.bind(on_release=lambda btn: popup.dismiss())
        content.add_widget(back_button)

        popup.open()

    def trigonometric_functions(self):
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(text="Enter the trigonometric function (e.g., sin(30), cos(45), tan(60), Note: After The Answer Is Giving Approximate It If Needed):"))
        expression_input = TextInput()
        content.add_widget(expression_input)

        popup = Popup(title='Trigonometric Functions', content=content, size_hint=(None, None), size=(1000, 400),
                      auto_dismiss=False)

        def on_calculate(instance):
            expression = expression_input.text
            if expression:
                try:
                    result = eval(expression, {'__builtins__': None},
                                  {'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'radians': math.radians})
                    result_label.text = f"The result is: {result}"
                except Exception as e:
                    result_label.text = f"Error: {str(e)}"

        result_label = Label(text="")
        calculate_button = Button(text="Calculate")
        calculate_button.bind(on_release=on_calculate)
        content.add_widget(result_label)
        content.add_widget(calculate_button)

        # Add back button
        back_button = Button(text="Back to Main Menu", size_hint=(None, None), size=(200, 40))
        back_button.bind(on_release=lambda btn: popup.dismiss())
        content.add_widget(back_button)

        popup.open()

    def matrix_inversion(self):
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(text="Enter the matrix as comma-separated rows (e.g., [[1, 2], [3, 4]]):"))
        matrix_input = TextInput()
        content.add_widget(matrix_input)

        popup = Popup(title='Matrix Inversion', content=content, size_hint=(None, None), size=(600, 400),
                      auto_dismiss=False)

        def on_invert(instance):
            matrix_str = matrix_input.text
            if matrix_str:
                try:
                    matrix = eval(matrix_str)
                    result = matrix_inversion(matrix)
                    result_label.text = f"The inverted matrix is: {result}"
                except Exception as e:
                    result_label.text = f"Error: {str(e)}"

        result_label = Label(text="")
        invert_button = Button(text="Invert")
        invert_button.bind(on_release=on_invert)
        content.add_widget(result_label)
        content.add_widget(invert_button)

        # Add back button
        back_button = Button(text="Back to Main Menu", size_hint=(None, None), size=(200, 40))
        back_button.bind(on_release=lambda btn: popup.dismiss())
        content.add_widget(back_button)

        popup.open()

    def matrix_determinant(self):
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(text="Enter the matrix as comma-separated rows (e.g., [[1, 2], [3, 4]]):"))
        matrix_input = TextInput()
        content.add_widget(matrix_input)

        popup = Popup(title='Matrix Determinant', content=content, size_hint=(None, None), size=(600, 400),
                      auto_dismiss=False)

        def on_calculate(instance):
            matrix_str = matrix_input.text
            if matrix_str:
                try:
                    matrix = eval(matrix_str)
                    result = matrix_determinant(matrix)
                    result_label.text = f"The determinant is: {result}"
                except Exception as e:
                    result_label.text = f"Error: {str(e)}"

        result_label = Label(text="")
        calculate_button = Button(text="Calculate")
        calculate_button.bind(on_release=on_calculate)
        content.add_widget(result_label)
        content.add_widget(calculate_button)

        # Add back button
        back_button = Button(text="Back to Main Menu", size_hint=(None, None), size=(200, 40))
        back_button.bind(on_release=lambda btn: popup.dismiss())
        content.add_widget(back_button)

        popup.open()

    def display_about(self, instance):
        about_info = (
            "DestiSolver AI\n\n"
            "A smart math problem-solving application.\n\n"
            "Created by Engineer Destiny Akhere and Engineer Success Mackson as a student-developed innovation for Mudiame University Irrua.\n\n"
            "Contact Information:\n"
            "Email: Destisolverai@gmail.com\n"
            "Phone Numbers: 09068754630, 0808 740 5076\n\n"
            "Follow us on Social Media:\n"
            "Facebook: @Destisolverai\n"
            "Twitter: @Destisolverai\n"
            "Instagram: @Destisolverai\n"
            "LinkedIn: @Destisolverai\n\n"
            "Copyright ©️ 2024")
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(text=about_info))

        back_button = Button(text="Back to Main Menu", size_hint=(None, None), size=(200, 40))
        back_button.bind(on_release=lambda btn: self.dismiss_popup())
        content.add_widget(back_button)

        popup = Popup(title='About DestiSolver AI', content=content, size_hint=(None, None), size=(400, 400))
        self._popup = popup  # Store the reference to the popup to access it later
        popup.open()

    def dismiss_popup(self):
        self._popup.dismiss()

    def exit_app(self, instance):
        App.get_running_app().stop()

    def general_answer(self):
        content = BoxLayout(orientation='vertical', padding=10)
        content.add_widget(Label(text="Describe your problem:"))
        problem_input = TextInput()
        content.add_widget(problem_input)

        popup = Popup(title='Describe Your Problem', content=content, size_hint=(None, None), size=(400, 200),
                      auto_dismiss=False)

        def on_submit(instance):
            problem_description = problem_input.text
            if not problem_description:
                return
            self.send_feedback(problem_description)
            # Update or remove messagebox based on your implementation
            messagebox.showinfo("Problem Description", f"Problem: {problem_description}\nFeedback submitted.")

        submit_button = Button(text="Submit")
        submit_button.bind(on_release=on_submit)
        content.add_widget(submit_button)

        popup.open()

    import smtplib
    from email.message import EmailMessage

    class DestiSolverApp(App):
        def build(self):
            return MyGrid()

        def send_feedback(self, feedback):
            msg = EmailMessage()
            msg.set_content(feedback)

            msg['Subject'] = 'Feedback from DestiSolver AI User'
            msg['From'] = 'destinyakhere98@gmail.com'  # Replace with your email address
            msg['To'] = 'Destisolverai@gmail.com'

            try:
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login('destinyakhere98@gmail.com',
                             'DestinyDestiny123456789@#')  # Replace with your email login credentials
                server.send_message(msg)
                server.quit()
                return True
            except Exception as e:
                print("Error sending feedback email:", e)
                return False

if __name__ == "__main__":
    app = DestiSolverApp()
    app.run()