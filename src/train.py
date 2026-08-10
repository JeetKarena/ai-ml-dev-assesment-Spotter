"""Training CLI entrypoint for Spotter."""

from src.pipelines.training_pipeline import TrainingPipeline


def main():
    """Run the training orchestration pipeline."""
    return TrainingPipeline().run()


if __name__ == "__main__":
    main()
