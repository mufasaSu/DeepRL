from .cql_agent import CQLAgent
from .dqn_agent import DQNAgent


agents = {
    "dqn": DQNAgent,
    "cql": CQLAgent,
}
