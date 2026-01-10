class ClusteringAlgRegister:
    alg_dict = {}

    @classmethod
    def register(cls, clustering_alg):
        cls.alg_dict[clustering_alg.__name__] = clustering_alg
        return clustering_alg

    @classmethod
    def get_subclasses(cls):
        return list(cls.alg_dict.keys())

    @classmethod
    def create_clustering_alg(cls, clustering_alg):
        if clustering_alg not in cls.alg_dict:
            raise ValueError(f"clustering algorithm {clustering_alg} not be registered")
        return cls.alg_dict[clustering_alg]