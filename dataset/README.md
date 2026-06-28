# Dataset folder

Place one reference image per person here. The filename stem becomes the label shown on screen(prototype) .

## Supported formats

`.jpg` `.jpeg` `.png` `.bmp` `.webp`

## Example layout

```
dataset/
    rishi.jpg
    alice.png
    bob.jpeg
```

## Tips

- Use a **well-lit, frontal** photo for best accuracy
- Filenames must not contain spaces — use underscores: `john_doe.jpg`
- The label shown will be exactly the filename stem (e.g. `rishi`)
