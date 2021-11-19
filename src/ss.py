from pprint import pprint
import numpy as np
import json
import time
import sobol_seq
import warnings
from typing import List
import pygmo as pg
import datetime
from search.moea_control import MOEActr
from generator import SamplesGenerator
from composite import ModelsUnion
from sklearn.model_selection import train_test_split
from hypothesis.custom_gp_kernel import KERNEL_MAUNA
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, RANSACRegressor
import sklearn.gaussian_process as gp
from ss_p import Preprocessing

warnings.filterwarnings("ignore")

def load_json_file(path_to_file):
    """
    Method reads .json file
    :param path_to_file: sting path to file.
    :return: object that represent .json file
    """
    with open(path_to_file, 'r') as File:
        jsonFile = json.loads(File.read())
        return jsonFile

def energy_consumption(task: dict):
    from random import choice
    from splitter import Splitter

    data = Splitter('Radix-500mio_avg.csv')
    data.search(str(task['Frequency']), str(task['Threads']))
    result = choice(data.new_data)
    return [float(result["EN"]), float(result['TIM'])]

def make_nd_pop(pro, x, y):
    nd_front = pg.fast_non_dominated_sorting(y)[0][0]
    nd_x = np.array(x)[nd_front]
    nd_y = np.array(y)[nd_front]
    t_pop = pg.population(pro)
    for i, p_vector in enumerate(nd_x):
        t_pop.push_back(x=p_vector, f=nd_y[i])
    return t_pop

class Problem:
    def __init__(self, search_space, regions, objectives) -> None:
        self.objectives = objectives
        self.search_space = search_space

        # Tuple of lower, upper bounds
        self.bounds = ([], [])
        for region, value in self.search_space.regions.items():
            if region in regions:
                for _, hyperparameter_values in value.items():
                    # Add lower bounds to first tuple, upper - to second
                    self.bounds[0].append(0)
                    self.bounds[1].append(1)

    def fitness(self, x):
        res = energy_consumption(x)
        return res[0]
    
    def get_bounds(self):
        return self.bounds
    
    def get_nobj(self):
        return self.objectives
    
class SearchSpace:
    def __init__(self, description: dict) -> None:
        self.description = description['Context']['Experiment']['SearchSpace']
        self.regions = {}
        self.regions_size = {}
        self.preprocessing = Preprocessing()
        # Keywords that cannot be defined for category name
        self.keywords = ['Type', 'Region', 'Size', 'Categories', 'Default', 'Lower', 'Upper', 'Bounds']

        # Initialize regions as dict with format {region_name: Dict}
        for reg in description['Context']['ExperimentRegions']['Regions']:
            self.regions.update({reg: {}})
        
        # Fill self.regions with data
        self.initialize_regions(self.description)

        # Count total number of features in the search space and its size
        self.features = 0
        self.size = 1
        for value in self.regions.values():
            self.features += len(value.keys())
            if self.size != np.inf:
                self.size *= self.get_region_size(value)

    def get_hyperparameter_size(self, hyperparameter):
        if hyperparameter['Type'] == 'IntegerHyperparameter':
            return hyperparameter['Upper'] - hyperparameter['Lower']
        elif hyperparameter['Type'] == 'FloatHyperparameter':
            return np.inf
        else:
            return len(hyperparameter['Categories'])
    
    def get_region_size(self, region):
        size = 1
        for value in region.values():
            size *= value['Size']
        return size

    def get_bounds(self, hyperparameter):
        if hyperparameter['Type'] == 'IntegerHyperparameter' or hyperparameter['Type'] == 'FloatHyperparameter':
            return [hyperparameter['Lower'], hyperparameter['Upper']]
        else:
            return [0, len(hyperparameter['Categories']) - 1]

    def initialize_regions(self, description: dict) -> None:
        for key, value in description.items():

            # Keyword objects are not count as features
            if key not in self.keywords:
                region_hp = self.regions[value['Region']]
                if value['Type'] != 'Category':
                    v_copy = {}
                    v_copy.update({'Size': self.get_hyperparameter_size(value)})
                    v_copy.update({'Bounds': self.get_bounds(value)})
                    # Remove parts which are related to other regions (not in keywords which describes this hyperparameter)
                    for k, v in value.items():
                        if k in self.keywords:
                            v_copy.update({k: v})

                    # Update self.regions
                    region_hp.update({key: v_copy})
                    self.regions.update({value['Region']: region_hp})

                # Recursion
                if type(value) is dict:
                    self.initialize_regions(value)
    
    def get_default_configuration(self):
        result = {'service_data': {'regions': {}}}
        index = 0

        # Beginning from TopLevel region
        regions_to_configure = ['TopLevel']

        # Before there are regions_to_configure
        while len(regions_to_configure) != 0:
            # For default configuration we pass vector as empty array
            reg_res, regions_to_configure, regions = self.configure_region(index, [], regions_to_configure, True)
            result.update(reg_res)
            result['service_data']['regions'].update(regions)
        return result

    def get_configuration(self, vector: List):
        result = {'service_data': {'regions': {}}}
        index = 0

        # Beginning from TopLevel region
        regions_to_configure = ['TopLevel']

        # Before there are regions_to_configure
        while len(regions_to_configure) != 0:
            reg_res, regions_to_configure, regions = self.configure_region(index, vector, regions_to_configure)
            result.update(reg_res)
            result['service_data']['regions'].update(regions)
        return result
    
    def configure_region(self, index, vector, region, default=False):
        result = {}
        regions = {}
        regions_to_configure = region
        i = index

        # Watch through first region in list
        for feature, value in self.regions[region[0]].items():
            # Adapt dimension number (useful in case of limited vector size - for example 40 for Sobol)
            if i > len(vector):
                i = i % len(vector)
            else:
                i = index
            
            # Denormalize vector component value in dependence on hyperparameter type
            if default is True:
                denorm_value = value['Default']
            else:
                denorm_value = self.denormalize(value, vector[i])
            result.update({feature: denorm_value})
            regions.update({feature: region[0]})
            index += 1

            # Add regions to check according to the selection in NominalHyperparameter
            if value['Type'] == 'NominalHyperparameter':
                if value['Region'] == 'TopLevel':
                    regions_to_configure.append(f'Context.Experiment.SearchSpace.{feature}.{denorm_value}')
                else:
                    regions_to_configure.append(f'{value["Region"]}.{feature}.{denorm_value}')
        # Remove checked region
        del region[0]
        return result, regions_to_configure, regions
    
    def denormalize(self, hyperparameter, vector_part):
        if hyperparameter['Type'] == 'IntegerHyperparameter':
            type = 'integer'
            bounds = hyperparameter
        elif hyperparameter['Type'] == 'FloatHyperparameter':
            type = 'float'
            bounds = hyperparameter
            return hyperparameter['Lower'] + vector_part * (hyperparameter['Upper'] - hyperparameter['Lower'])
        else:
            type = 'categorical'
            bounds = {'Lower': 0, 'Upper': len(hyperparameter['Categories']) - 1}
        res = self.preprocessing.denormalization(vector_part, type, bounds)
        if hyperparameter['Type'] not in ['IntegerHyperparameter', 'FloatHyperparameter']:
            return hyperparameter['Categories'][res]
        else:
            return res
    
    def find(self, key, value):
        for k, v in value.items():
            if k == key:
                yield v
            elif isinstance(v, dict):
                for result in self.find(key, v):
                    yield result

    def normalize(self, hyperparameter: dict):
        key, value = list(hyperparameter.items())[0]
        hyperparameter_object = list(self.find(key, self.regions))
        for match in hyperparameter_object:
            try:
                if match['Type'] not in ['IntegerHyperparameter', 'FloatHyperparameter']:
                    bounds = {'Lower': 0, 'Upper': len(match['Categories']) - 1}
                    value = match['Categories'].index(value)
                else:
                    bounds = match
                return self.preprocessing.normalization(value, bounds)
            except ValueError:
                pass
    
    def serialize(self):
        serialization = dict(
                size=self.size,
                name=self.name,
                boundaries=self.regions,
                root_parameters_list=self.regions['TopLevel'].keys()
            )
        return serialization


class Predictor:
    def __init__(self, search_space, surr_portfolio, region, udp, pro) -> None:
        self.preprocessing = Preprocessing()
        self.search_space = search_space
        self.surr_portfolio = surr_portfolio
        self.region = region
        self.udp = udp
        self.pro = pro
    
    def get_results(self):
        X, y = self.gen.return_X_y()
        return np.array(X), np.array(y)

    def tuning_loop(self, X_init, Y_init, predict=True):
        loop_start = time.time()
        iter_solution = []
        self.gen = SamplesGenerator(self.pro)
        # Evaluate initial set
        x_init = []
        y_init = []
        for index, configuration in enumerate(X_init):
            subres = []
            for hyperparameter, value in configuration.items():
                if hyperparameter != 'service_data' and configuration['service_data']['regions'][hyperparameter] in self.region:
                    subres.append(self.search_space.normalize({hyperparameter: value}))
            if predict is False:
                for _ in range(len(subres), len(self.udp.bounds[0])):
                    subres.append(0)
            if len(subres) == len(self.udp.bounds[0]):
                x_init.append(subres)
                y_init.append(Y_init[index])

        if np.array(x_init).size > 0:
            self.gen.update(x_init, y_init)

            pred = {}
            pred['iteration'] = 0
            pred['problem'] = self.pro.get_name()
            pred['objectives'] = self.pro.get_nobj()
            pred['feature_dim'] = self.pro.get_nx()
            try:
                # ref_point = pg.nadir(np.array(y_init))
                ref_point = np.amax(np.array(y_init), axis=0).tolist()
                pred['ref_point'] = ref_point
                nd_pop = make_nd_pop(self.pro, np.array(X_init), np.array(y_init))
                hypervolume = pg.hypervolume(nd_pop.get_f()
                                            ).compute(ref_point)
                pred['hypervolume'] = hypervolume or None
                pred["ndf_size"] = len(nd_pop.get_f())
                pred["i_fevals"] = self.pro.get_fevals()
                pred["pop_ndf_x"] = nd_pop.get_x().tolist()
                pred["pop_ndf_y"] = nd_pop.get_f().tolist()

                score = self.udp.p_distance(nd_pop) if hasattr(
                    self.udp, 'p_distance') else None
                pred["p_distance"] = score
                iter_solution.append(pred)
            except Exception as err:
                pred['error'] = "Init stat: {}".format(err)
                iter_solution.append(pred)
        else:
            return 'Sampling'
        if predict == False:
            return None
        X, y = self.gen.return_X_y()
        # equalize the number of samples for your portfolio and static models
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25)
        estimator = self.surr_portfolio[0].fit(X_train, y_train)
        if isinstance(estimator, ModelsUnion):
            models = estimator.models
        else:
            models = [estimator]

        solver = MOEActr(bounds=self.pro.get_bounds(),
                            pop_size=100, gen=100)
        solver.fit(models)
        appr = solver.predict()
        propos = appr[np.random.choice(
            appr.shape[0], 1, replace=False), :][0]
        return propos


class Orchectrator():
    def __init__(self) -> None:
        main = './SettingsBRISE.json'
        alternative = './SettingsBRISEcopy.json'

        descr = load_json_file(alternative)
        self.ss_class = SearchSpace(descr)
        for reg, value in self.ss_class.regions.items():
            print(f'Region: {reg}, Value: {value}')

        print(f'Total features number: {self.ss_class.features}')
        print(f'Search space size: {self.ss_class.size}')

        svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        svr_uni = ModelsUnion(models=[svr_rbf], split_y=True)
        surr_port = [
            [svr_uni],
            [LinearRegression()],
            [ModelsUnion(models=[LinearRegression()],
                            split_y=True)],
            [RANSACRegressor()],
            [ModelsUnion(models=[RANSACRegressor()],
                            split_y=True)],
            [ModelsUnion(models=[gp.GaussianProcessRegressor(
                kernel=KERNEL_MAUNA, n_restarts_optimizer=20)], split_y=True)],
            [gp.GaussianProcessRegressor(
                kernel=KERNEL_MAUNA, n_restarts_optimizer=20)]
        ]
        self.udp = Problem(self.ss_class, self.ss_class.regions, len(descr['Context']['Experiment']['TaskConfiguration']['Objectives'].keys()))
        self.prob = pg.problem(self.udp)
        print(self.prob)
        self.mode = descr['Context']['Model']['Structure']
        self.predictor = Predictor(self.ss_class, surr_port, self.ss_class.regions, self.udp, self.prob)
        if self.mode == 'Hierarchical':
            self.predictors = {}
            for index, region in enumerate(descr['Context']['ExperimentRegions']['Regions']):
                regions = {region: self.ss_class.regions[region]}
                udp = Problem(self.ss_class, regions, len(descr['Context']['Experiment']['TaskConfiguration']['Objectives'].keys()))
                prob = pg.problem(udp)
                print(prob)
                self.predictors.update({region: Predictor(self.ss_class, surr_port[index], [region], udp, prob)})

    def initial_configs(self):

        configuration_number = range(0, 10)
        self.seed = 0
        if self.ss_class.features > 40:
            self.dimensionality = 40
        else:
            self.dimensionality = self.ss_class.features
        self.features = []
        self.labels = []
        default = self.ss_class.get_default_configuration()
        print(f'Default Configuration: {default}')
        self.features.append(default)
        self.labels.append(energy_consumption(default))
        for _ in configuration_number:
            vector, self.seed = sobol_seq.i4_sobol(self.dimensionality, self.seed)
            print(f'Sobol vector: {vector}')
            config = self.ss_class.get_configuration(vector)
            print(f'Configuration: {config}')
            if config not in self.features:
                self.features.append(config)
                self.labels.append(energy_consumption(config))

        print(f'Sampled features: {self.features}')
        print(f'Their labels: {self.labels}')

    def prediction_part(self):
        evolve_start = time.time()
        for _ in range(0, 10):
            print('__________________________')
            if self.mode == 'Flat':
                prediction = self.predictor.tuning_loop(self.features, self.labels)
            else:
                prediction = []
                regions_to_configure = ['TopLevel']
                while len(regions_to_configure) != 0:
                    print(f'Predicting Region {regions_to_configure[0]}')
                    sub = self.predictors[regions_to_configure[0]].tuning_loop(self.features, self.labels)
                    if sub == 'Sampling':
                        vector, self.seed = sobol_seq.i4_sobol(self.predictors[regions_to_configure[0]].udp.objectives, self.seed)
                        sub = vector
                    _, regions_to_configure, _ = self.ss_class.configure_region(0, sub, regions_to_configure)
                    for elem in sub:
                        prediction.append(elem)
                self.predictor.tuning_loop(self.features, self.labels, False)
            prediction = self.ss_class.get_configuration(prediction)
            print(f'Prediction: {prediction}')
            while prediction in self.features:
                vector, self.seed = sobol_seq.i4_sobol(self.dimensionality, self.seed)
                prediction = self.ss_class.get_configuration(vector)
                if prediction not in self.features:
                    print(f'Sampled random config: {prediction}')
            self.features.append(prediction)
            print(f'Configuration result: {energy_consumption(prediction)}')
            self.labels.append(energy_consumption(prediction))

        result = {}
        try:
            x_loop, y_loop = self.predictor.get_results()
            result["fevals"] = self.prob.get_fevals()
            nd_pop = make_nd_pop(self.prob, x_loop, y_loop)
            score = self.udp.p_distance(nd_pop) if hasattr(self.udp, 'p_distance') else None
            result["p_distance"] = score or None
        except Exception as err:
            print(f'ERROR OCCURED!!! {err}')
            result['error'] = "Solo loop: {}".format(err)

        # ----------------------                                                            Hypervolume
        try:
            # ref_point = pg.nadir(y_loop)
            ref_point = np.amax(y_loop, axis=0).tolist()

            hypervolume = pg.hypervolume(nd_pop.get_f()
                                            ).compute(ref_point)
            result['hypervolume'] = hypervolume or None
        except Exception as err:
            result['error'] = "Hypervolume: {}".format(err)

        # ----------------------                                                            Spacing metric
        # The spacing metric aims at assessing the spread (distribution)
        # of vectors throughout the set of nondominated solutions.
        try:
            dist = pg.crowding_distance(points=nd_pop.get_f())
            not_inf_dist = dist[np.isfinite(dist)]
            mean_dist = np.mean(not_inf_dist)
            space_m = (sum([(mean_dist - d)**2 for d in not_inf_dist]
                            )/(len(not_inf_dist)-1))**(1/2)
            result["ndf_space"] = space_m
        except Exception as err:
            result['error'] = "Spacing metric: {}".format(err)

        # ----------------------                                                            Write results
        try:
            t_end = time.time()

            result["pop_ndf_x"] = nd_pop.get_x().tolist()
            for index in range(0, len(result['pop_ndf_x'])):
                result['pop_ndf_x'][index] = self.ss_class.get_configuration(result['pop_ndf_x'][index])
            result["pop_ndf_f"] = nd_pop.get_f().tolist()
            result["ndf_size"] = len(nd_pop.get_f())
            result["evolve_time"] = t_end - evolve_start
            result["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            result["final"] = True

        except Exception as err:
            result['error'] = "Write results: {}".format(err)

        print(f'____________________')
        print(f'Final results')

        pprint(f'Sampled features: {self.features}')
        pprint(f'Their labels: {self.labels}')

        print(f'____________________')
        print(f'Solution')
        for index, solution in enumerate(result['pop_ndf_f']):
            print(f'Solution {index}. Configuration: {self.features[self.labels.index(solution)]}')
            print(f'Solution {index}. Result: {solution}')


if __name__ == "__main__":
    module = Orchectrator()
    module.initial_configs()
    module.prediction_part()