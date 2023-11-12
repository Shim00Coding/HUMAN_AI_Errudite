from typing import List, Dict, Any
from ..utils import Registrable

class Predictor(Registrable):
    """A base class for predictors.
    A predictor runs prediction on raw texts and also instances.
    It also saves the performance score for the predictor.
        
    This is a subclass of ``errudite.utils.registrable.Registrable`` and all the actual rewrite 
    rule classes are registered under ``Predictor`` by their names.

    Parameters
    ----------
    name : str
        The name of the predictor.
    description : str
        A sentence describing the predictor.
    model : any
        The executable model.
    perform_metrics : List[str]
        The name of performance metrics.
    
    Attributes
    ----------
    perform : Dict[str, float]
        .. code-block:: js
            
            { perform_name: the averaged performance score. }
    """
    def __init__(self, 
        name: str, 
        description: str, 
        model: any,
        perform_metrics: List[str]):
        self.name: str = name
        self.description: str = description
        self.predictor: Any = model
        self.perform: Dict[str, float] = {}
        self.perform_metrics: List[str] = perform_metrics

        for p in self.perform_metrics:
            self.perform[p] = 0
    
    def predict(self, **kwargs):
        """
        run the prediction.

        Raises
        ------
        NotImplementedError
           Should be implemented in subclasses.
        """
        raise NotImplementedError

    def evaluate_performance(self, instances: List['Instance']) -> None:
        """Save the performance of the predictor.
        It iterates through metric names in ``self.perform_metrics``, and average the 
        corresponding metrics in ``instance.prediction.perform``. It saves the results
        in ``self.perform``.
        
        Parameters
        ----------
        instances : List[Instance]
            The list of instances, with predictions from this model already saved as
            part of its entries.
        
        Returns
        -------
        None
            The result is saved in ``self.perform``.
        """
        instances = list(filter(lambda i: i.vid==0, instances))
        n_total = len(instances)
        if n_total != 0:
            for metric in self.perform_metrics:
                if metric == 'accuracy':
                    count_correct_predictions = 0
                    for i in instances:
                        pe_accuracy = i.get_entry('prediction_PE', self.name).perform['accuracy']
                        ke_accuracy = i.get_entry('prediction_KE', self.name).perform['accuracy']
                        lce_accuracy = i.get_entry('prediction_LCE', self.name).perform['accuracy']
                        
                        if pe_accuracy == 1 and ke_accuracy == 1 and lce_accuracy == 1:
                            count_correct_predictions += 1

                    self.perform[metric] = count_correct_predictions / n_total
                #PE 
                if metric == 'accuracy_PE_Acceptable':
                    acceptable_pe = 0
                    acceptable_num = 0
                    for i in instances:
                        value = i.get_entry('prediction_PE', self.name).label
                        if(value == 'Acceptable'):
                            acceptable_num += 1
                            if(value == i.get_entry('groundtruth_PE', self.name).label):
                                acceptable_pe += 1
                    if (acceptable_num == 0):
                        self.perform[metric] = 'None'
                    else:
                        self.perform[metric] = acceptable_pe / acceptable_num
                if metric == 'accuracy_PE_Unacceptable':
                    unacceptable_pe = 0
                    unacceptable_num = 0
                    for i in instances:
                        value = i.get_entry('prediction_PE', self.name).label
                        if(value == 'Unacceptable'):
                            unacceptable_num += 1
                            if(value == i.get_entry('groundtruth_PE', self.name).label):
                                unacceptable_pe += 1
                    if (unacceptable_num == 0):
                        self.perform[metric] = 'None'
                    else:
                        self.perform[metric] = unacceptable_pe / unacceptable_num
                if metric == 'accuracy_PE_Insufficient':
                    insufficient_pe = 0
                    insufficient_num = 0
                    for i in instances:
                        value = i.get_entry('prediction_PE', self.name).label
                        if(value == 'Insufficient'):
                            insufficient_num += 1
                            if(value == i.get_entry('groundtruth_PE', self.name).label):
                                insufficient_pe += 1
                    if (insufficient_num == 0):
                        self.perform[metric] = 'None'
                    else:
                        self.perform[metric] = insufficient_pe / insufficient_num
                if metric == 'accuracy_PE_NotFound':
                    notfound_pe = 0
                    notfound_num = 0
                    for i in instances:
                        value = i.get_entry('prediction_PE', self.name).label
                        if(value == 'Not Found'):
                            notfound_num += 1
                            if(value == i.get_entry('groundtruth_PE', self.name).label):
                                notfound_pe += 1
                    if (notfound_num == 0):
                        self.perform[metric] = 'None'
                    else:
                        self.perform[metric] = notfound_pe / notfound_num
                #KE
                if metric == 'accuracy_KE_Acceptable':
                    acceptable_ke = 0
                    acceptable_num = 0
                    for i in instances:
                        value = i.get_entry('prediction_KE', self.name).label
                        if(value == 'Acceptable'):
                            acceptable_num += 1
                            if(value == i.get_entry('groundtruth_KE', self.name).label):
                                acceptable_ke += 1
                    if (acceptable_num == 0):
                        self.perform[metric] = 'None'
                    else:
                        self.perform[metric] = acceptable_ke / acceptable_num
                if metric == 'accuracy_KE_Unacceptable':
                    unacceptable_ke = 0
                    unacceptable_num = 0
                    for i in instances:
                        value = i.get_entry('prediction_KE', self.name).label
                        if(value == 'Unacceptable'):
                            unacceptable_num += 1
                            if(value == i.get_entry('groundtruth_KE', self.name).label):
                                unacceptable_ke += 1
                    if (unacceptable_num == 0):
                        self.perform[metric] = 'None'
                    else:
                        self.perform[metric] = unacceptable_ke / unacceptable_num
                if metric == 'accuracy_KE_Insufficient':
                    insufficient_ke = 0
                    insufficient_num = 0
                    for i in instances:
                        value = i.get_entry('prediction_KE', self.name).label
                        if(value == 'Insufficient'):
                            insufficient_num += 1
                            if(value == i.get_entry('groundtruth_KE', self.name).label):
                                insufficient_ke += 1
                    if (insufficient_num == 0):
                        self.perform[metric] = 'None'
                    else:
                        self.perform[metric] = insufficient_ke / insufficient_num
                if metric == 'accuracy_KE_NotFound':
                    notfound_ke = 0
                    notfound_num = 0
                    for i in instances:
                        value = i.get_entry('prediction_KE', self.name).label
                        if(value == 'Not Found'):
                            notfound_num += 1
                            if(value == i.get_entry('groundtruth_KE', self.name).label):
                                notfound_ke += 1
                    if (notfound_num == 0):
                        self.perform[metric] = 'None'
                    else:
                        self.perform[metric] = notfound_ke / notfound_num
                ##LCE
                if metric == 'accuracy_LCE_Acceptable':
                    acceptable_lce = 0
                    acceptable_num = 0
                    for i in instances:
                        value = i.get_entry('prediction_LCE', self.name).label
                        if(value == 'Acceptable'):
                            acceptable_num += 1
                            if(value == i.get_entry('groundtruth_LCE', self.name).label):
                                acceptable_lce += 1
                    if (acceptable_num == 0):
                        self.perform[metric] = 'None'
                    else:
                        self.perform[metric] = acceptable_lce / acceptable_num
                if metric == 'accuracy_LCE_Unacceptable':
                    unacceptable_lce = 0
                    unacceptable_num = 0
                    for i in instances:
                        value = i.get_entry('prediction_LCE', self.name).label
                        if(value == 'Unacceptable'):
                            unacceptable_num += 1
                            if(value == i.get_entry('groundtruth_LCE', self.name).label):
                                unacceptable_lce += 1
                    if (unacceptable_num == 0):
                        self.perform[metric] = 'None'
                    else:
                        self.perform[metric] = unacceptable_lce / unacceptable_num
                if metric == 'accuracy_LCE_Insufficient':
                    insufficient_lce = 0
                    insufficient_num = 0
                    for i in instances:
                        value = i.get_entry('prediction_LCE', self.name).label
                        if(value == 'Insufficient'):
                            insufficient_num += 1
                            if(value == i.get_entry('groundtruth_LCE', self.name).label):
                                insufficient_lce += 1
                    if (insufficient_num == 0):
                        self.perform[metric] = 'None'
                    else:
                        self.perform[metric] = insufficient_lce / insufficient_num
                if metric == 'accuracy_LCE_NotFound':
                    notfound_lce = 0
                    notfound_num = 0
                    for i in instances:
                        value = i.get_entry('prediction_KE', self.name).label
                        if(value == 'Not Found'):
                            notfound_num += 1
                            if(value == i.get_entry('groundtruth_LCE', self.name).label):
                                notfound_lce += 1
                    if (notfound_num == 0):
                        self.perform[metric] = 'None'
                    else:
                        self.perform[metric] = notfound_lce / notfound_num
        else:
            print(n_total)
            print(self.name)
    
    def serialize(self) -> Dict:
        """Seralize the instance into a json format, for sending over
        to the frontend.
        
        Returns
        -------
        Dict[str, Any]
            The serialized version.
        """
        return {
            'perform': self.perform,
            'name': self.name,
            'description': self.description
        }
    
    def __repr__(self) -> str:
        """
        Override the print func by displaying the class name and the predictor name.
        """
        return f'{self.__class__.__name__} {self.name}'
    
    @classmethod
    def create_from_json(cls, raw: Dict[str, str]) -> 'Predictor':
        """
        Recreate the predictor from its seralized raw json.
        
        Parameters
        ----------
        raw : Dict[str, str]
            The json version definition of the predictor, with 
            name, description, model_path, and model_online_path.

        Returns
        -------
        Predictor
            The re-created predictor.
        """
        try:
            return Predictor.by_name(raw["model_class"])(
                name=raw["name"] if "name" in raw else None, 
                description=raw["description"] if "description" in raw else None,
                model_path=raw["model_path"] if "model_path" in raw else None,
                model_online_path=raw["model_online_path"] if "model_online_path" in raw else None)
        except:
            raise
    
    @classmethod
    def model_predict(cls, 
        predictor: 'Predictor', 
        **targets) -> 'Label':
        """
        Define a class method that takes Target inputs, run model predictions, 
        and wrap the output prediction into Labels.
        
        Parameters
        ----------
        predictor : Predictor
            A predictor object, with the predict method implemented.
        targets : Target
            Targets in kwargs format

        Returns
        -------
        Label
            The predicted output, with performance saved.
        
        Raises
        -------
        NotImplementedError
            This needs to be implemented per task.
        """
        raise NotImplementedError