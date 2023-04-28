from tfne.encodings.codeepneat.codeepneat_blueprint import CoDeepNEATBlueprint

class LamaCoDeepNEATBlueprint(CoDeepNEATBlueprint):
    """
    LamaCoDeepNEATBlueprint inherits from CoDeepNEATBlueprint.
    Following methods are overwritten:
    * __init__
    * __str__
    Following methods are new:
    * set_flops
    * get_flops
    * get_rank
    * get_crowd_distance
    * set_id
    * get_siblings
    * set_siblings
    """

    def __init__(self,
                 blueprint_id,
                 parent_mutation,
                 blueprint_graph,
                 optimizer_factory):
        # Register parameters
        self.blueprint_id = blueprint_id
        self.parent_mutation = parent_mutation
        self.blueprint_graph = blueprint_graph
        self.optimizer_factory = optimizer_factory

        # Initialize internal variables
        self.fitness = 100
        self.flops = 0
        self.dominates = []
        self.rank = -1
        self.dom_counter = 0
        self.distance = 0
        self.siblings = 1

        # Declare graph related internal variables
        # species: set of all species present in blueprint
        # node_species: dict mapping of each node to its corresponding species
        # node dependencies: dict mapping of nodes to the set of node upon which they depend upon
        # graph topology: list of sets of dependency levels, with the first set being the nodes that depend on nothing,
        #                 the second set being the nodes that depend on the first set, and so on
        self.species = set()
        self.node_species = dict()
        self.node_dependencies = dict()
        self.graph_topology = list()

        # Process graph to set graph related internal variables
        self._process_graph()

    def __str__(self) -> str:
        """
        @return: string representation of the blueprint
        """
        return "LamaCoDeepNEAT Blueprint | ID: {:>6} | Fitness: {:>6} | FLOPs: {} | Nodes: {:>4} | Module Species: {} | " \
               "Optimizer: {}" \
            .format('#' + str(self.blueprint_id),
                    self.fitness,
                    self.flops,
                    len(self.node_species),
                    self.species,
                    self.optimizer_factory.get_name())

    def get_flops(self):
        return self.flops

    def set_flops(self, flops):
        self.flops = flops

    def get_crowd_distance(self):
        return self.distance

    def get_rank(self):
        return self.rank

    def set_id(self, bp_id):
        self.blueprint_id = bp_id

    def get_siblings(self):
        return self.siblings

    def set_siblings(self, siblings):
        self.siblings = siblings



