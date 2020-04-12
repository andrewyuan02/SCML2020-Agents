
class AgentProfile:
    def __init__(self):
        self.id = None
        self.initial_balance = None

        self.input_product = None
        self.output_product = None

        self.n_lines = None  # production capacity
        self.n_processes = None  # n_products - 1
        self.n_products = None
        self.n_steps = None

        self.supplier_list = None
        self.consumer_list = None

        self.supplier_matrix = None  # first index is product
        self.consumer_matrix = None  # first index is product
        self.catalog_price_list = None

        self.process = None  # is equal to input product
        self.production_cost = None

        self.agents = None

        self.last_day = None


class AgentPlan:
    def __init__(self):
        self.target_input = None  # How much input I want to possess at each step
        self.expected_input = None
        self.min_sell_price = None
        self.available_money = None
        self.available_output = None
        self.true_input = None


class BuyPlan:
    def __init__(self, target_input, production_capacity, contract_inputs=0, excess_inputs=0):
        self.target_input = target_input
        self.production_capacity = production_capacity
        self.contract_inputs = contract_inputs  # inputs received from that day
        self.excess_inputs = excess_inputs  # inputs which are left unused from previous day

    def get_total(self):
        return self.contract_inputs + self.excess_inputs

    def get_needed(self):
        return max(self.target_input - self.get_total(), 0)  # Non-negative

    def get_excess(self):  # inputs which could be used tomorrow
        return max(self.get_total() - self.production_capacity, 0)  # Assumes we can always afford production cost
