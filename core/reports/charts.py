import logging
from typing import Optional

from .schema import SecurityReport, severity_counts, framework_counts

logger = logging.getLogger(__name__)


def render_severity_chart(report: SecurityReport, output_path: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = severity_counts(report)
        severities = list(counts.keys())
        values = list(counts.values())
        colors = ["#8B0000", "#C0392B", "#E67E22", "#218738", "#5D6D7E"]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(severities, values, color=colors)
        ax.set_title("Findings by Severity")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return output_path
    except Exception as e:
        logger.warning("Failed to render severity chart: %s", e)
        return None
    finally:
        if "fig" in locals():
            plt.close(fig)


def render_framework_chart(report: SecurityReport, output_path: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = framework_counts(report)
        if not counts or all(v == 0 for v in counts.values()):
            return None

        frameworks = list(counts.keys())
        values = list(counts.values())

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(frameworks, values, color="#0B1F3A")
        ax.set_title("Findings by Framework")
        ax.set_xlabel("Count")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return output_path
    except Exception as e:
        logger.warning("Failed to render framework chart: %s", e)
        return None
    finally:
        if "fig" in locals():
            plt.close(fig)
