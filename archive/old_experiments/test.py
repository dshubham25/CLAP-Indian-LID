import requests
from bs4 import BeautifulSoup


def decode_message(url):
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    points = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")

        if not rows:
            continue
        headers = [cell.get_text(" ", strip=True).lower()
                   for cell in rows[0].find_all(["td", "th"])]

        if not {"x-coordinate", "character", "y-coordinate"}.issubset(headers):
            continue

        x_idx = headers.index("x-coordinate")
        char_idx = headers.index("character")
        y_idx = headers.index("y-coordinate")

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])

            if len(cells) <= max(x_idx, char_idx, y_idx):
                continue

            try:
                x = int(cells[x_idx].get_text(strip=True))
                y = int(cells[y_idx].get_text(strip=True))
                char = cells[char_idx].get_text("", strip=True)

                if char:
                    points.append((x, y, char))
            except ValueError:
                continue

        break

    if not points:
        raise ValueError("No Coordination")
    max_x = max(x for x, _, _ in points)
    max_y = max(y for _, y, _ in points)
    grid = {(x, y): char for x, y, char in points}
    for y in range(max_y, -1, -1):
        row = []
        for x in range(max_x + 1):
            row.append(grid.get((x, y), " "))
        print("".join(row))
if __name__ == "__main__":
    decode_message(
        "https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"
    )