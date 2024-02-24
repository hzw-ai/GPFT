import warnings

warnings.filterwarnings('ignore')

import sys

sys.path.append('.')
sys.path.append('..')
import yaml
import argparse
import traceback
import time
import torch

from model.models import STSSL
from model.graph_prompt import SimplePrompt, GPFplusAtt
from model.trainer_p import Trainer
from lib.dataloader import get_dataloader
from lib.utils import (
    init_seed,
    get_model_params,
    load_graph,
)


def model_supervisor(args):
    init_seed(args.seed)
    if not torch.cuda.is_available():
        args.device = 'cpu'

    ## load dataset
    dataloader = get_dataloader(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
    )
    graph = load_graph(args.graph_file, device=args.device)
    args.num_nodes = len(graph)

    ## init model and set optimizer
    model = STSSL(args).to(args.device)

    state_dict = torch.load(
        f'experiments/{args.datasets}/best_model.pth',
        map_location=torch.device(args.device)
    )
    model.load_state_dict(state_dict['model'])
    model.requires_grad_(False)

    if args.prompt == 'SimplePrompt':
        # prompt = SimplePrompt(2, dtype=torch.float32).to(args.device)
        prompt = SimplePrompt(args.d_model, dtype=torch.float32).to(args.device)
        prompt_2 = SimplePrompt(args.d_model, dtype=torch.float32).to(args.device)
    else:
        # prompt = GPFplusAtt(2, p_num=args.p_num, dtype=torch.float32).to(args.device)
        prompt = GPFplusAtt(args.d_model, p_num=args.p_num, dtype=torch.float32).to(args.device)
        prompt_2 = GPFplusAtt(args.d_model, p_num=args.p_num, dtype=torch.float32).to(args.device)

    # model_parameters = get_model_params([model])
    model_parameters = []
    model_parameters.append({"params": prompt.parameters(), "lr": args.lr_init_p})
    model_parameters.append({"params": prompt_2.parameters(), "lr": args.lr_init_p})
    # model.mlp.requires_grad_(True)
    # model_parameters.append({"params": model.mlp.parameters()})
    # model.encoder.pooler.requires_grad_(True)
    # model_parameters.append({"params": model.encoder.pooler.parameters()})
    optimizer = torch.optim.Adam(
        params=model_parameters,
        lr=args.lr_init,
        eps=1.0e-8,
        weight_decay=0,
        amsgrad=False
    )

    ## start training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        graph=graph,
        args=args,
        prompt=prompt,
        prompt_2=prompt_2
    )
    results = None
    try:
        if args.mode == 'train':
            results = trainer.train()  # best_eval_loss, best_epoch
        elif args.mode == 'test':
            # test
            print("Load saved model")
            results = trainer.test(model, dataloader['test'], dataloader['scaler'],
                                   graph, trainer.logger, trainer.args, prompt, prompt_2)
        else:
            raise ValueError
    except:
        trainer.logger.info(traceback.format_exc())
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', default='NYCTaxi', type=str, help='the configuration to use')
    # parser.add_argument('--config_filename', default='configs/NYCBike1.yaml',
    #                     type=str, help='the configuration to use')
    parser.add_argument('--prompt', default='GPFplusAtt', choices=['GPFplusAtt', 'SimplePrompt'], type=str)
    parser.add_argument('--p_num', default=8, type=int)
    args = parser.parse_args()

    print(f'Starting experiment with configurations in configs/{args.datasets}.yaml...')
    time.sleep(3)
    configs = yaml.load(
        open(f'configs/{args.datasets}.yaml'),
        Loader=yaml.FullLoader
    )
    # experiments/NYCBike1/20240125-160229/best_model.pth

    args = argparse.Namespace(**{**args.__dict__, **configs})
    # args = argparse.Namespace(**{**args, **configs})
    model_supervisor(args)
