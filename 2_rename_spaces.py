from pathlib import Path

ROOT = Path("./PlantVillage")
DRY_RUN = False


def sanitize(name: str) -> str:
    return name.replace(" ", "_")


def main():
    if not ROOT.is_dir():
        print(f"Folder not found: {ROOT.resolve()}")
        return

    renamed_files = 0
    renamed_dirs = 0

    for path in sorted(ROOT.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if " " not in path.name:
            continue

        new_path = path.with_name(sanitize(path.name))

        if new_path.exists():
            print(f"Skip (target exists): {path}")
            continue

        if path.is_dir():
            renamed_dirs += 1
        else:
            renamed_files += 1

        print(f"{path}  ->  {new_path}")

        if not DRY_RUN:
            path.rename(new_path)

    print("\n" + "=" * 60)
    if DRY_RUN:
        print("DRY RUN - nothing was changed")
    print(f"Renamed entries: {renamed_files + renamed_dirs}")
    print(f"  folders: {renamed_dirs}")
    print(f"  files:   {renamed_files}")


if __name__ == "__main__":
    main()