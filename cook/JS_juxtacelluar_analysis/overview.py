
from kitchen.loader.ephys_loader import cohort_loader


if __name__ == "__main__":
    x = cohort_loader("PassivePuff_JuxtaCelluar_FromJS", "SST_EXAMPLE")
    print(x)