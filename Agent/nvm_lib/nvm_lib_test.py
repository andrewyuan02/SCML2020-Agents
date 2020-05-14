from nvm_lib import NVMLib

# Needs to be called just once in the constructor of the agent
nvm = NVMLib(mpnvp_number_of_periods=4,
             mpnvp_quantities_domain_size=20,
             game_length=10,
             input_product_index=0,
             output_product_index=1,
             num_intermediate_products=1,
             production_cost=1.0)

# Need to be called at each simulation time. Returns the plan for step current_time. Verbose to get some info into what is going on.
nvm_sol = nvm.get_complete_plan(current_time=0,
                                verbose=True)
print(f'nvm_sol = {nvm_sol}')
