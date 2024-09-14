def join_paths(*paths: str) -> str:
    """
        joins the path of two path objects. strips any excess '/' chars from inputs before joining with '/'
    """

    cleaned_paths = [path.strip('/') for path in paths]
    return "/".join(cleaned_paths)
