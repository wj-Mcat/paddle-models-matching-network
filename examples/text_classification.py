from src.config import Config
from src.data_processors import TNewsDataProcessor, DataProcessor
from trainer import Trainer

def run():
    config = Config().parse_args(known_only=True)
    processor = TNewsDataProcessor(config.data_dir, '_0')
    matching_network = 
    trainer = Trainer(
        config=config,
        processor=processor,
        
    )