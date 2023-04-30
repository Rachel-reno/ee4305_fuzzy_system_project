# ee4305_fuzzy_system_project
Health prediction using fuzzy logic system with data collected from wearables

1. Packages needed 

   For all the codes numpy, matplotlib and argparse is needed. While for the skfuzzy implementation a scikit-fuzzy package need to be downloaded that can be done using `pip install scikit-fuzzy`
   
2. How to run the files

   To run the files, download the files to your computer. The `fuzzy_logic_code.py` is a basic implementation of fuzzy logic without external packages. You will be able to plot the membership function by uncommenting the `plot_fuzzy_sets` function and adjusting the parameters. To run the prediction run this line of code in the terminal `python fuzzy_logic_code.py --age 65 --hr 130`. The value 65 and 130 can be changed according to the parameters that will be inputted. To run the code using the skfuzzy run this line of code `python fuzzy_logic_code_skfuzzy.py --age 65 --hr 130 --show_mf True --show_decision True`. There are two additional parameters that will take in boolean values. Input True for show_mf if you would like to visualize the fuzzy sets of the inputs and output and input True for show_decision if you would like to visualize the output result in fuzzy sets. Similarly, to run the extension of the algorithm that includes the activity level, run this line of code `python fuzzy_system_w_activity.py --age 65 --hr 130 --activity 0.8 --show_mf True --show_decision True` and change it according to the inputs. 
   
3. Debugging

   If the code does not work in the terminal. Comment from the line `if __name__ == '__main__'` to the end of the file and simply call the function `health_prediction()` with the selected input. 
