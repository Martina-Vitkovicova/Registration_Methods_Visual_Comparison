FILEPATH = "C:\\Users\\vitko\\Desktop\\ProjetHCI-BT\\BT_implementation\\Organs\\"

PATIENTS = ["137", "146", "148", "198", "489", "579", "716", "722"]
TIMESTAMPS = list(range(1, 14))

# organ traces: prostate, bones, bladder, rectum
BLUE1 = "#008698"
BLUE2 = "#4D5FEB"
BLUE3 = "#94D2FF"
BLUE4 = "#439FFF"

# custom heatmap colour scale
HEATMAP_CS = [[0.00, "#01494D"],
              [0.24, "#01766A"],
              [0.38, "#0d8f81"],
              [0.50, "#39ab7e"],
              [0.66, "#6ec574"],
              [0.83, "#a9dc67"],
              [1.00, "#edef5d"]]

# axes colours
GREEN = "#23D495"
YELLOW = "#F0E736"
ORANGE = "#FF8725"

# timestamp colours
PURPLE = "#824AB2"
PINK = "#E6598D"

# others
GREY = "#838AA3"
GREY2 = "#46495A"
LIGHT_GREY = "#BFE0EE"
LIGHT_GREY2 = "#696E80"
LIGHT_GREEN = "#C3EFD9"

# evaluation colours
ev1 = "#8CDBFF"
ev2 = "#53BEEE"
ev3 = "#1E9BD4"
ev4 = "#147FB3"
ev5 = "#0C527C"
DARK_GREEN = "#139C79"
DARK_YELLOW = "#ECB823"
ACT_RED = "#EC6923"
DARK_RED = "#D52F0C"

CONE_TIP = 0.1
CONE_START = 0.9

# which function we want to get rid of in the modebar
MODEBAR = ["autoScale2d", "lasso2d", "select2d", "zoomOut2d", "zoomIn2d"]
D3_MODEBAR = ["orbitRotation", "resetCameraDefault3d"]
