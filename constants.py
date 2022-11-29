FILEPATH = "C:\\Users\\vitko\\Desktop\\ProjetHCI-BT\\BT_implementation\\Organs\\"

PATIENTS = ["137", "146", "148", "198", "489", "579", "716", "722"]
TIMESTAMPS = list(range(1, 14))
scale = {"137": 39, "146": 73, "148": 55, "198": 79, "489": 92, "579": 28, "716": 85, "722": 57}

LIGHT_BLUE = "#008698"  # blue
PURPLE = "#4D5FEB"  # purple
GREEN = "#94D2FF"  # orange
RED = "#439FFF"  # pink
CYAN = "#22CCA2"  # cyan

# LIGHT_BLUE = "#2896CF"  # blue
# PURPLE = "#9742D0"  # purple
# GREEN = "#F4E236"  # yellow
# RED = "#E35649"  # red
# CYAN = "#22CCA2"  # cyan

HEATMAP_CS = [[0.00, "#01494D"],
              [0.24, "#01766A"],
              [0.38, "#0d8f81"],
              [0.50, "#39ab7e"],
              [0.66, "#6ec574"],
              [0.83, "#a9dc67"],
              [1.00, "#edef5d"]]

CYAN1 = "#E89E08"
CYAN2 = "#F9E806"
CYAN3 = "#97D820"

GREY2 = "#555868"
LIGHT_GREY2 = "#767B8D"
GREY = "#838AA3"
LIGHT_GREY = "#BFE0EE"
LIGHT_GREEN = "#C3EFD9"

ORANGE = "#824AB2"  # orange
PINK = "#E6598D"  # pink

# CYAN1 = "#333338"
# CYAN2 = "#595a62"
# CYAN3 = "#D9DAE2"

# CYAN1 = "#0E5342"
# CYAN2 = "#189A7A"
# CYAN3 = "#A7F6D9"

# GREEN2 = "#A6FB3D"  # green
# ORANGE = "#F8AD24"  # orange
# PINK = "#E6598D"  # pink
# GREY = "#838AA3"
# LIGHT_GREY = "#AEB3C3"
# LIGHT_PURPLE = "#CCCCFF"

COLORSCALE = [[0.00, '#1A3770'],
              [0.12, '#124F84'],
              [0.30, '#107C94'],
              [0.50, '#3DDA86'],
              [0.70, '#84F15D'],
              [1.00, '#F0F63A']]

COLORSCALE_BLACK = [[0.00, '#333338'],
                    # [0.12, '#383838'],
                    [0.30, '#63646D'],
                    # [0.50, '#878686'],
                    # [0.70, '#AAA9A9'],
                    [1.00, '#D9DAE2']]

COLORSCALE_BLUE = [[0.00, '#1A3770'],
                   # [0.12, '#0159AA'],
                   [0.30, '#1C6FCB'],
                   # [0.50, '#0778B9'],
                   # [0.70, '#338BC5'],
                   [1.00, '#A3E4FF']]

COLORSCALE_YL = [[0.00, '#383838'],
                 # [0.20, '#4A3A2A'],
                 # [0.30, '#704b25'],
                 [0.40, '#FD9223'],
                 [0.70, '#FDD523'],
                 [1.00, '#FFFF93']]

COLORSCALE_GRN = [[0.00, '#13593F'],
                  # [0.20, '#4A3A2A'],
                  # [0.30, '#704b25'],
                  # [0.40, '#FD9223'],
                  # [0.70, '#FDD523'],
                  [1.00, '#AEEED6']]

COLORSCALE_PNK = [[0.00, '#4D0226'],
                  # [0.20, '#4A3A2A'],
                  [0.20, '#7E1548'],
                  [0.45, '#AF2166'],
                  [0.70, '#E58614'],
                  [1.00, '#F2A50B']]

COLORSCALE_ORN = [[0.00, '#4D2102'],
                  # [0.20, '#4A3A2A'],
                  # [0.20, '#7E1548'],
                  # [0.45, '#AF2166'],
                  # [0.70, '#E58614'],
                  [1.00, '#F19511']]

CONE_TIP = 0.1
CONE_START = 0.9

MODEBAR = ["autoScale2d", "lasso2d", "toImage", "select2d", "zoomOut2d", "zoomIn2d"]

D3_MODEBAR = ["toImage", "orbitRotation", "resetCameraDefault3d"]
