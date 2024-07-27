# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 09:38:00 2023

@author: marco
"""

import sympy as sp

# def preprocess_formula(expression):
#     return expression.replace("^2", "**2").replace("^", "**")
 
# The dynamically created functions from add_features_gui will be written here

# def A_feature(b, c):
#     var_dict = {var: float(val) for var, val in zip(independent_vars, args)}
#     processed_expression = preprocess_formula(expression)
#     return float(sp.sympify(processed_expression, locals=var_dict))

# def P_feature(t, n):
#     var_dict = {var: float(val) for var, val in zip(independent_vars, args)}
#     processed_expression = preprocess_formula(expression)
#     return float(sp.sympify(processed_expression, locals=var_dict))

# import sympy as sp
# def preprocess_formula(expression):
#     return expression.replace("^2", "**2").replace("^", "**")
# def M_feature(n, x, expression):
#     var_dict = {var: float(val) for var, val in zip(independent_vars, args)}
#     processed_expression = preprocess_formula(expression)
#     return float(sp.sympify(processed_expression, locals=var_dict))

# import sympy as sp
# def preprocess_formula(expression):
#     return expression.replace("^2", "**2").replace("^", "**")
# def Q_feature(a, p, expression):
#     var_dict = {var: float(val) for var, val in zip([a, p], [a, p])}
#     processed_expression = preprocess_formula(expression)
#     return float(sp.sympify(processed_expression, locals=var_dict))

# import sympy as sp
# def preprocess_formula(expression):
#     return expression.replace("^2", "**2").replace("^", "**")
# def A_feature(t, p, x, expression):
#     print("Here 1")
#     var_dict = {var: float(val) for var, val in zip([t, p, x], [t, p, x])}
#     print("Here 2")
#     processed_expression = preprocess_formula(expression)
#     print("Here 3")
#     return float(sp.sympify(processed_expression, locals=var_dict))

# import sympy as sp
# def preprocess_formula(expression):
#     return expression.replace("^2", "**2").replace("^", "**")
# def P_feature(a, b, c, expression):
#     var_dict = {var: float(val) for var, val in zip([a, b, c], [a, b, c])}
#     processed_expression = preprocess_formula(expression)
#     return float(sp.sympify(processed_expression, locals=var_dict))

# import sympy as sp
# def preprocess_formula(expression):
#     return expression.replace("^2", "**2").replace("^", "**")
# def X_feature(t, y, u, expression):
#     var_dict = {var: float(val) for var, val in zip([t, y, u], [t, y, u])}
#     processed_expression = preprocess_formula(expression)
#     return float(sp.sympify(processed_expression, locals=var_dict))

# import sympy as sp
# def preprocess_formula(expression):
#     return expression.replace("^2", "**2").replace("^", "**")
# def L_feature(s, p, f, d, expression):
#     var_dict = {var: float(val) for var, val in zip([s, p, f, d], [s, p, f, d])}
#     processed_expression = preprocess_formula(expression)
#     return float(sp.sympify(processed_expression, locals=var_dict))

# import sympy as sp
# def preprocess_formula(expression):
#     return expression.replace("^2", "**2").replace("^", "**")
# def I_feature(t, p, x, expression):
#     var_dict = {var: float(val) for var, val in zip([t, p, x], [t, p, x])}
#     processed_expression = preprocess_formula(expression)
#     return float(sp.sympify(processed_expression, locals=var_dict))

# import sympy as sp
# def preprocess_formula(expression):
#     return expression.replace("^2", "**2").replace("^", "**")
# def H_feature(a, s, d, p, expression):
#     var_dict = {var: float(val) for var, val in zip([a, s, d, p], [a, s, d, p])}
#     processed_expression = preprocess_formula(expression)
#     return float(sp.sympify(processed_expression, locals=var_dict))

# import sympy as sp
# def preprocess_formula(expression):
#     return expression.replace("^2", "**2").replace("^", "**")
# def X_feature(a, b, c, expression):
#     var_dict = {var: float(val) for var, val in zip([a, b, c], [a, b, c])}
#     processed_expression = preprocess_formula(expression)
#     return float(sp.sympify(processed_expression, locals=var_dict))

# import sympy as sp
# def preprocess_formula(expression):
#     return expression.replace("^2", "**2").replace("^", "**")
# def G_feature(w, e, r, t, expression):
#     var_dict = {var: float(val) for var, val in zip([w, e, r, t], [w, e, r, t])}
#     processed_expression = preprocess_formula(expression)
#     return float(sp.sympify(processed_expression, locals=var_dict))

# import sympy as sp
# def preprocess_formula(expression):
#     return expression.replace("^2", "**2").replace("^", "**")
# def F_feature(a, s, g, expression):
#     var_dict = {var: float(val) for var, val in zip([a, s, g], [a, s, g])}
#     processed_expression = preprocess_formula(expression)
#     return float(sp.sympify(processed_expression, locals=var_dict))

# import sympy as sp
# def preprocess_formula(expression):
#     return expression.replace("^2", "**2").replace("^", "**")
# def B_feature(a, t, o, expression):
#     var_dict = {var: float(val) for var, val in zip([a, t, o], [a, t, o])}
#     processed_expression = preprocess_formula(expression)
#     return float(sp.sympify(processed_expression, locals=var_dict))

# import sympy as sp
# def preprocess_formula(expression):
#     return expression.replace("^2", "**2").replace("^", "**")
# def K_feature(a, w, e, expression):
#     var_dict = {var: float(val) for var, val in zip([a, w, e], [a, w, e])}
#     processed_expression = preprocess_formula(expression)
#     return float(sp.sympify(processed_expression, locals=var_dict))

# import sympy as sp
# def preprocess_formula(expression):
#     return expression.replace("^2", "**2").replace("^", "**")
# def O_feature(a, b, c, d, expression):
#     print("-- Here")
#     var_dict = {var: float(val) for var, val in zip([a, b, c, d], [a, b, c, d])}
#     print("A")
#     processed_expression = preprocess_formula(expression)
#     print("B")
#     print("A: " + str(processed_expression) + " ...")
#     print("result")
#     ind_eq = 0
#     for indP,p in enumerate(processed_expression):
#         if p == '=':
#             ind_eq = indP
#             break
        
#     processed_expression = processed_expression[(ind_eq+1):]
#     print(str(sp.sympify(processed_expression, locals=var_dict)))
#     return float(sp.sympify(processed_expression, locals=var_dict))
 
# import sympy as sp
# def preprocess_formula(expression):
#  #   print("Result of preprocess: " + str(expression.replace("^", "**")))
# #    return str(expression.replace("^", "**"))
#     return "E = a**s+d*p"
# def E_feature(a, s, d, p, expression):
#     print("--Here")
#     var_dict = {var: float(val) for var, val in zip([a, s, d, p], [a, s, d, p])}
#     print("--Here 2")
#     expression.replace("^", "**")
#     print("--Here 3")
#     processed_expression = expression
#     print(processed_expression)
    
#     print(str(eval("a**s+d*p")))
# #    processed_expression = preprocess_formula(expression)
#     print("Result: " + str(eval(processed_expression)))
#     # import sys
#     # sys.exit()
#     return eval(processed_expression) 

import sympy as sp
def preprocess_formula(expression):
    return expression.replace("^2", "**2").replace("^", "**")
def Y_feature(a, b, c, d, expression):
    print("A")
    var_dict = {var: float(val) for var, val in zip([a, b, c, d], [a, b, c, d])}
    processed_expression = preprocess_formula(expression)   
    return eval(processed_expression)

import sympy as sp
def T_feature(a, x, d, f, expression):
    print("A")
    var_dict = {var: float(val) for var, val in zip([a, x, d, f], [a, x, d, f])}
    expression.replace("^", "**")
    print("B")
    print(expression)
    processed_expression = expression
    print(eval(processed_expression)) 
    return eval(processed_expression)

import sympy as sp
def J_feature(a, m, o, r, expression):
    var_dict = {var: float(val) for var, val in zip([a, m, o, r], [a, m, o, r])}
    expression.replace("^", "**")
    processed_expression = expression
    return eval(processed_expression)

import sympy as sp
def C_feature(a, d, f, g, h, expression):
    var_dict = {var: float(val) for var, val in zip([a, d, f, g, h], [a, d, f, g, h])}
    expression.replace("^", "**")
    processed_expression = expression
    return eval(processed_expression)

import sympy as sp
def R_feature(a, s, d, expression):
    var_dict = {var: float(val) for var, val in zip([a, s, d], [a, s, d])}
    expression.replace("^", "**")
    processed_expression = expression
    return eval(processed_expression)

import sympy as sp
def W_feature(a, c, d, expression):
    var_dict = {var: float(val) for var, val in zip([a, c, d], [a, c, d])}
    expression.replace("^", "**")
    processed_expression = expression
    return eval(processed_expression)

import sympy as sp
def Z_feature(a, s, d, expression):
    var_dict = {var: float(val) for var, val in zip([a, s, d], [a, s, d])}
    expression.replace("^", "**")
    processed_expression = expression
    return eval(processed_expression)

import sympy as sp
def V_feature(a, d, f, t, expression):
    var_dict = {var: float(val) for var, val in zip([a, d, f, t], [a, d, f, t])}
    expression.replace("^", "**")
    processed_expression = expression
    return eval(processed_expression)

import sympy as sp
def A_feature(a, s, f, r, expression):
    var_dict = {var: float(val) for var, val in zip([a, s, f, r], [a, s, f, r])}
    expression.replace("^", "**")
    processed_expression = expression
    return eval(processed_expression)

import sympy as sp 
def W_feature(q, r, e, expression):
    var_dict = {var: float(val) for var, val in zip([q, r, e], [q, r, e])}
    expression.replace("^", "**")
    processed_expression = expression
    return eval(processed_expression)

import sympy as sp
def S_feature(a, d, f, expression):
    var_dict = {var: float(val) for var, val in zip([a, d, f], [a, d, f])}
    expression.replace("^", "**")
    processed_expression = expression
    return eval(processed_expression)

import sympy as sp
def B_feature(a, s, d, expression):
    var_dict = {var: float(val) for var, val in zip([a, s, d], [a, s, d])}
    expression.replace("^", "**")
    processed_expression = expression
    return eval(processed_expression)

