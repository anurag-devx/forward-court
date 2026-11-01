"""Run the gateway: python -m gateway (uvicorn :8000)."""
import uvicorn


def main() -> None:
    uvicorn.run("gateway.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
