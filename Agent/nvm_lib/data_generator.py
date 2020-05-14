import pandas as pd
import data

game_length = 10
num_intermediate_products = 1

"""
A synthetic example of what the game's log might look like. 
The NVM code needs the game log's to be a dataframe where each row has at least
    Time, Product, Quantity, Price
Any other column is ignored. 
The interpretation of a row is a transaction that occur between to agents at 
simulation Time for Product for Quantity at Price (unit price). 
"""
# TODO explain this example.
dummy_game_log = pd.DataFrame([{'time': 1, 'product': 'p0', 'quantity': 1, 'price': 1110.23},
                               {'time': 0, 'product': 'p0', 'quantity': 1, 'price': 3.5},
                               {'time': 2, 'product': 'p1', 'quantity': 1, 'price': 1.23},
                               {'time': 2, 'product': 'p1', 'quantity': 5, 'price': 1.8},
                               {'time': 3, 'product': 'p1', 'quantity': 5, 'price': 1.8}],
                              columns=['time', 'product', 'quantity', 'price'], index=[0, 1, 2, 3, 4])

# Replace game_logs here with actual game logs.
game_logs = dummy_game_log

"""
Save quantity uncertainty model.
The model is saved to json file data/dict_qtty_num_intermediate_products_{num_intermediate_products}.json
The model is read as product -> time -> quantity -> probability of observing the quantity of the product traded at the time. 
"""
data.save_json_qytt_uncertainty_model(json_file_name=f'dict_qtty_num_intermediate_products_{num_intermediate_products}',
                                      the_game_logs=game_logs,
                                      game_length=game_length)

"""
Save price uncertainty model.
The model is saved to json file data/dict_price_num_intermediate_products_{num_intermediate_products}.json
The model is read as product -> time -> average price at which the product was traded at the time.
"""
data.save_json_price_data(json_file_name=f'dict_price_num_intermediate_products_{num_intermediate_products}',
                          the_game_logs=game_logs)
