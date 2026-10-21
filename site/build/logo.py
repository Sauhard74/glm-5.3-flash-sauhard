"""Write the logo: public/favicon.svg, and build/out-icon.html (a 512 px render to screenshot into PNG icons).

The mark is three isometric plates, the layer stack from Fig 1, with the top plate lit.
"""

from pathlib import Path

HERE = Path(__file__).parent


def plate(y, top, left, right):
    return (f'<path d="M14 {y}L32 {y + 9}L32 {y + 13}L14 {y + 4}Z" fill="{left}"/>'
            f'<path d="M32 {y + 9}L50 {y}L50 {y + 4}L32 {y + 13}Z" fill="{right}"/>'
            f'<path d="M14 {y}L32 {y - 9}L50 {y}L32 {y + 9}Z" fill="{top}"/>')


BODY = (plate(41, "#3b3b42", "#232328", "#2c2c32") + plate(33, "#4a4a52", "#2a2a30", "#35353c")
        + plate(25, "#7b9dff", "#3f5bc4", "#5577e6"))
SVG = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="#0b0b0c"/>{BODY}</svg>'

if __name__ == "__main__":
    (HERE.parent / "public" / "favicon.svg").write_text(SVG)
    big = SVG.replace("<svg ", '<svg width="512" height="512" ', 1)
    (HERE / "out-icon.html").write_text(
        f'<!doctype html><html><body style="margin:0"><div id="i" style="width:512px;height:512px">{big}</div></body></html>')
    print(SVG)
