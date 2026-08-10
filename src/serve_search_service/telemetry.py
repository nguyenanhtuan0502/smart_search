"""OpenTelemetry resource setup for the search service."""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider


def configure_observability(*, service_name: str, service_version: str, environment: str) -> None:
    """Set a tracer resource without replacing SDK/autoinstrumentation providers."""
    if trace.get_tracer_provider().__class__.__name__ == "ProxyTracerProvider":
        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": service_name,
                    "service.version": service_version,
                    "deployment.environment": environment,
                }
            )
        )
        trace.set_tracer_provider(provider)
