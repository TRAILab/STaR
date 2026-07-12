from dataclasses import dataclass, field
import inspect

@dataclass
class AgentOutput:
    type: str
    text: str
    binary: str
    position: list
    orientation: float
    duration: float
    time: float
    object_id: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, dict_input):      
        return cls(**{
            k: v for k, v in dict_input.items() 
            if k in inspect.signature(cls).parameters
        })


class Agent:
    def query(self, query: str) -> AgentOutput:
        raise NotImplementedError

    def query_position(self, query: str) -> list:
        return self.query(query).position
    
    def query_duration(self, query: str) -> float:
        return self.query(query).duration

    def query_time(self, query: str) -> float:
        return self.query(query).time
    
    def query_yes_no(self, query: str) -> bool:
        str_bool = self.query(query).binary
        if str_bool.lower() == 'yes':
            return True
        return False
