from app import create_app


def main():
    app = create_app()

    # dev запуск
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )


if __name__ == "__main__":
    main()
