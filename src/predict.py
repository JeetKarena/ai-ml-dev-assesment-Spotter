"""Inference CLI entrypoint for Spotter."""

from src.pipelines.inference_pipeline import InferencePipeline


def main(payload=None):
    """Run the inference orchestration pipeline."""
    return InferencePipeline().run(payload)


if __name__ == "__main__":
    main()
