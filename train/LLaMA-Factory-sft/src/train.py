
from llamafactory.train.tuner import run_exp


def main():
    run_exp()


def _mp_fn(index):
    run_exp()


if __name__ == "__main__":
    main()
