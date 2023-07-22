import glob
from copy import deepcopy
import os
from os.path import join
import PIL
from PIL import Image
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.functional import resize as vis_resize
from tqdm import tqdm

UTK_ATTRS = ['age', 'gender', 'race']

PROMPTS = {'utk': "f'a person of age {c}.'",
           'adience': "f'age {c}'",
           'smallnorb':
               {'azimuth': "f'an object facing azimuth {c}'"},
           'stanford_cars': "f'a car from {c}'",
           'cifar10': "f'a photo of a {c}'",
           'imagenet': "f'a photo of a {c}'"
           }

CIFAR10_CLASS_LABELS = ["airplane",
                        "automobile",
                        "bird",
                        "cat",
                        "deer",
                        "dog",
                        "frog",
                        "horse",
                        "ship",
                        "truck"]

IMAGENET_CLASS_LABELS = ["tench",
                         "goldfish",
                         "great white shark",
                         "tiger shark",
                         "hammerhead shark",
                         "electric ray",
                         "stingray",
                         "cock",
                         "hen",
                         "ostrich",
                         "brambling",
                         "goldfinch",
                         "house finch",
                         "junco",
                         "indigo bunting",
                         "American robin",
                         "bulbul",
                         "jay",
                         "magpie",
                         "chickadee",
                         "American dipper",
                         "kite",
                         "bald eagle",
                         "vulture",
                         "great grey owl",
                         "fire salamander",
                         "smooth newt",
                         "newt",
                         "spotted salamander",
                         "axolotl",
                         "American bullfrog",
                         "tree frog",
                         "tailed frog",
                         "loggerhead sea turtle",
                         "leatherback sea turtle",
                         "mud turtle",
                         "terrapin",
                         "box turtle",
                         "banded gecko",
                         "green iguana",
                         "Carolina anole",
                         "desert grassland whiptail lizard",
                         "agama",
                         "frilled-necked lizard",
                         "alligator lizard",
                         "Gila monster",
                         "European green lizard",
                         "chameleon",
                         "Komodo dragon",
                         "Nile crocodile",
                         "American alligator",
                         "triceratops",
                         "worm snake",
                         "ring-necked snake",
                         "eastern hog-nosed snake",
                         "smooth green snake",
                         "kingsnake",
                         "garter snake",
                         "water snake",
                         "vine snake",
                         "night snake",
                         "boa constrictor",
                         "African rock python",
                         "Indian cobra",
                         "green mamba",
                         "sea snake",
                         "Saharan horned viper",
                         "eastern diamondback rattlesnake",
                         "sidewinder",
                         "trilobite",
                         "harvestman",
                         "scorpion",
                         "yellow garden spider",
                         "barn spider",
                         "European garden spider",
                         "southern black widow",
                         "tarantula",
                         "wolf spider",
                         "tick",
                         "centipede",
                         "black grouse",
                         "ptarmigan",
                         "ruffed grouse",
                         "prairie grouse",
                         "peacock",
                         "quail",
                         "partridge",
                         "grey parrot",
                         "macaw",
                         "sulphur-crested cockatoo",
                         "lorikeet",
                         "coucal",
                         "bee eater",
                         "hornbill",
                         "hummingbird",
                         "jacamar",
                         "toucan",
                         "duck",
                         "red-breasted merganser",
                         "goose",
                         "black swan",
                         "tusker",
                         "echidna",
                         "platypus",
                         "wallaby",
                         "koala",
                         "wombat",
                         "jellyfish",
                         "sea anemone",
                         "brain coral",
                         "flatworm",
                         "nematode",
                         "conch",
                         "snail",
                         "slug",
                         "sea slug",
                         "chiton",
                         "chambered nautilus",
                         "Dungeness crab",
                         "rock crab",
                         "fiddler crab",
                         "red king crab",
                         "American lobster",
                         "spiny lobster",
                         "crayfish",
                         "hermit crab",
                         "isopod",
                         "white stork",
                         "black stork",
                         "spoonbill",
                         "flamingo",
                         "little blue heron",
                         "great egret",
                         "bittern",
                         "crane (bird)",
                         "limpkin",
                         "common gallinule",
                         "American coot",
                         "bustard",
                         "ruddy turnstone",
                         "dunlin",
                         "common redshank",
                         "dowitcher",
                         "oystercatcher",
                         "pelican",
                         "king penguin",
                         "albatross",
                         "grey whale",
                         "killer whale",
                         "dugong",
                         "sea lion",
                         "Chihuahua",
                         "Japanese Chin",
                         "Maltese",
                         "Pekingese",
                         "Shih Tzu",
                         "King Charles Spaniel",
                         "Papillon",
                         "toy terrier",
                         "Rhodesian Ridgeback",
                         "Afghan Hound",
                         "Basset Hound",
                         "Beagle",
                         "Bloodhound",
                         "Bluetick Coonhound",
                         "Black and Tan Coonhound",
                         "Treeing Walker Coonhound",
                         "English foxhound",
                         "Redbone Coonhound",
                         "borzoi",
                         "Irish Wolfhound",
                         "Italian Greyhound",
                         "Whippet",
                         "Ibizan Hound",
                         "Norwegian Elkhound",
                         "Otterhound",
                         "Saluki",
                         "Scottish Deerhound",
                         "Weimaraner",
                         "Staffordshire Bull Terrier",
                         "American Staffordshire Terrier",
                         "Bedlington Terrier",
                         "Border Terrier",
                         "Kerry Blue Terrier",
                         "Irish Terrier",
                         "Norfolk Terrier",
                         "Norwich Terrier",
                         "Yorkshire Terrier",
                         "Wire Fox Terrier",
                         "Lakeland Terrier",
                         "Sealyham Terrier",
                         "Airedale Terrier",
                         "Cairn Terrier",
                         "Australian Terrier",
                         "Dandie Dinmont Terrier",
                         "Boston Terrier",
                         "Miniature Schnauzer",
                         "Giant Schnauzer",
                         "Standard Schnauzer",
                         "Scottish Terrier",
                         "Tibetan Terrier",
                         "Australian Silky Terrier",
                         "Soft-coated Wheaten Terrier",
                         "West Highland White Terrier",
                         "Lhasa Apso",
                         "Flat-Coated Retriever",
                         "Curly-coated Retriever",
                         "Golden Retriever",
                         "Labrador Retriever",
                         "Chesapeake Bay Retriever",
                         "German Shorthaired Pointer",
                         "Vizsla",
                         "English Setter",
                         "Irish Setter",
                         "Gordon Setter",
                         "Brittany",
                         "Clumber Spaniel",
                         "English Springer Spaniel",
                         "Welsh Springer Spaniel",
                         "Cocker Spaniels",
                         "Sussex Spaniel",
                         "Irish Water Spaniel",
                         "Kuvasz",
                         "Schipperke",
                         "Groenendael",
                         "Malinois",
                         "Briard",
                         "Australian Kelpie",
                         "Komondor",
                         "Old English Sheepdog",
                         "Shetland Sheepdog",
                         "collie",
                         "Border Collie",
                         "Bouvier des Flandres",
                         "Rottweiler",
                         "German Shepherd Dog",
                         "Dobermann",
                         "Miniature Pinscher",
                         "Greater Swiss Mountain Dog",
                         "Bernese Mountain Dog",
                         "Appenzeller Sennenhund",
                         "Entlebucher Sennenhund",
                         "Boxer",
                         "Bullmastiff",
                         "Tibetan Mastiff",
                         "French Bulldog",
                         "Great Dane",
                         "St. Bernard",
                         "husky",
                         "Alaskan Malamute",
                         "Siberian Husky",
                         "Dalmatian",
                         "Affenpinscher",
                         "Basenji",
                         "pug",
                         "Leonberger",
                         "Newfoundland",
                         "Pyrenean Mountain Dog",
                         "Samoyed",
                         "Pomeranian",
                         "Chow Chow",
                         "Keeshond",
                         "Griffon Bruxellois",
                         "Pembroke Welsh Corgi",
                         "Cardigan Welsh Corgi",
                         "Toy Poodle",
                         "Miniature Poodle",
                         "Standard Poodle",
                         "Mexican hairless dog",
                         "grey wolf",
                         "Alaskan tundra wolf",
                         "red wolf",
                         "coyote",
                         "dingo",
                         "dhole",
                         "African wild dog",
                         "hyena",
                         "red fox",
                         "kit fox",
                         "Arctic fox",
                         "grey fox",
                         "tabby cat",
                         "tiger cat",
                         "Persian cat",
                         "Siamese cat",
                         "Egyptian Mau",
                         "cougar",
                         "lynx",
                         "leopard",
                         "snow leopard",
                         "jaguar",
                         "lion",
                         "tiger",
                         "cheetah",
                         "brown bear",
                         "American black bear",
                         "polar bear",
                         "sloth bear",
                         "mongoose",
                         "meerkat",
                         "tiger beetle",
                         "ladybug",
                         "ground beetle",
                         "longhorn beetle",
                         "leaf beetle",
                         "dung beetle",
                         "rhinoceros beetle",
                         "weevil",
                         "fly",
                         "bee",
                         "ant",
                         "grasshopper",
                         "cricket",
                         "stick insect",
                         "cockroach",
                         "mantis",
                         "cicada",
                         "leafhopper",
                         "lacewing",
                         "dragonfly",
                         "damselfly",
                         "red admiral",
                         "ringlet",
                         "monarch butterfly",
                         "small white",
                         "sulphur butterfly",
                         "gossamer-winged butterfly",
                         "starfish",
                         "sea urchin",
                         "sea cucumber",
                         "cottontail rabbit",
                         "hare",
                         "Angora rabbit",
                         "hamster",
                         "porcupine",
                         "fox squirrel",
                         "marmot",
                         "beaver",
                         "guinea pig",
                         "common sorrel",
                         "zebra",
                         "pig",
                         "wild boar",
                         "warthog",
                         "hippopotamus",
                         "ox",
                         "water buffalo",
                         "bison",
                         "ram",
                         "bighorn sheep",
                         "Alpine ibex",
                         "hartebeest",
                         "impala",
                         "gazelle",
                         "dromedary",
                         "llama",
                         "weasel",
                         "mink",
                         "European polecat",
                         "black-footed ferret",
                         "otter",
                         "skunk",
                         "badger",
                         "armadillo",
                         "three-toed sloth",
                         "orangutan",
                         "gorilla",
                         "chimpanzee",
                         "gibbon",
                         "siamang",
                         "guenon",
                         "patas monkey",
                         "baboon",
                         "macaque",
                         "langur",
                         "black-and-white colobus",
                         "proboscis monkey",
                         "marmoset",
                         "white-headed capuchin",
                         "howler monkey",
                         "titi",
                         "Geoffroy's spider monkey",
                         "common squirrel monkey",
                         "ring-tailed lemur",
                         "indri",
                         "Asian elephant",
                         "African bush elephant",
                         "red panda",
                         "giant panda",
                         "snoek",
                         "eel",
                         "coho salmon",
                         "rock beauty",
                         "clownfish",
                         "sturgeon",
                         "garfish",
                         "lionfish",
                         "pufferfish",
                         "abacus",
                         "abaya",
                         "academic gown",
                         "accordion",
                         "acoustic guitar",
                         "aircraft carrier",
                         "airliner",
                         "airship",
                         "altar",
                         "ambulance",
                         "amphibious vehicle",
                         "analog clock",
                         "apiary",
                         "apron",
                         "waste container",
                         "assault rifle",
                         "backpack",
                         "bakery",
                         "balance beam",
                         "balloon",
                         "ballpoint pen",
                         "Band-Aid",
                         "banjo",
                         "baluster",
                         "barbell",
                         "barber chair",
                         "barbershop",
                         "barn",
                         "barometer",
                         "barrel",
                         "wheelbarrow",
                         "baseball",
                         "basketball",
                         "bassinet",
                         "bassoon",
                         "swimming cap",
                         "bath towel",
                         "bathtub",
                         "station wagon",
                         "lighthouse",
                         "beaker",
                         "military cap",
                         "beer bottle",
                         "beer glass",
                         "bell-cot",
                         "bib",
                         "tandem bicycle",
                         "bikini",
                         "ring binder",
                         "binoculars",
                         "birdhouse",
                         "boathouse",
                         "bobsleigh",
                         "bolo tie",
                         "poke bonnet",
                         "bookcase",
                         "bookstore",
                         "bottle cap",
                         "bow",
                         "bow tie",
                         "brass",
                         "bra",
                         "breakwater",
                         "breastplate",
                         "broom",
                         "bucket",
                         "buckle",
                         "bulletproof vest",
                         "high-speed train",
                         "butcher shop",
                         "taxicab",
                         "cauldron",
                         "candle",
                         "cannon",
                         "canoe",
                         "can opener",
                         "cardigan",
                         "car mirror",
                         "carousel",
                         "tool kit",
                         "carton",
                         "car wheel",
                         "automated teller machine",
                         "cassette",
                         "cassette player",
                         "castle",
                         "catamaran",
                         "CD player",
                         "cello",
                         "mobile phone",
                         "chain",
                         "chain-link fence",
                         "chain mail",
                         "chainsaw",
                         "chest",
                         "chiffonier",
                         "chime",
                         "china cabinet",
                         "Christmas stocking",
                         "church",
                         "movie theater",
                         "cleaver",
                         "cliff dwelling",
                         "cloak",
                         "clogs",
                         "cocktail shaker",
                         "coffee mug",
                         "coffeemaker",
                         "coil",
                         "combination lock",
                         "computer keyboard",
                         "confectionery store",
                         "container ship",
                         "convertible",
                         "corkscrew",
                         "cornet",
                         "cowboy boot",
                         "cowboy hat",
                         "cradle",
                         "crane (machine)",
                         "crash helmet",
                         "crate",
                         "infant bed",
                         "Crock Pot",
                         "croquet ball",
                         "crutch",
                         "cuirass",
                         "dam",
                         "desk",
                         "desktop computer",
                         "rotary dial telephone",
                         "diaper",
                         "digital clock",
                         "digital watch",
                         "dining table",
                         "dishcloth",
                         "dishwasher",
                         "disc brake",
                         "dock",
                         "dog sled",
                         "dome",
                         "doormat",
                         "drilling rig",
                         "drum",
                         "drumstick",
                         "dumbbell",
                         "Dutch oven",
                         "electric fan",
                         "electric guitar",
                         "electric locomotive",
                         "entertainment center",
                         "envelope",
                         "espresso machine",
                         "face powder",
                         "feather boa",
                         "filing cabinet",
                         "fireboat",
                         "fire engine",
                         "fire screen sheet",
                         "flagpole",
                         "flute",
                         "folding chair",
                         "football helmet",
                         "forklift",
                         "fountain",
                         "fountain pen",
                         "four-poster bed",
                         "freight car",
                         "French horn",
                         "frying pan",
                         "fur coat",
                         "garbage truck",
                         "gas mask",
                         "gas pump",
                         "goblet",
                         "go-kart",
                         "golf ball",
                         "golf cart",
                         "gondola",
                         "gong",
                         "gown",
                         "grand piano",
                         "greenhouse",
                         "grille",
                         "grocery store",
                         "guillotine",
                         "barrette",
                         "hair spray",
                         "half-track",
                         "hammer",
                         "hamper",
                         "hair dryer",
                         "hand-held computer",
                         "handkerchief",
                         "hard disk drive",
                         "harmonica",
                         "harp",
                         "harvester",
                         "hatchet",
                         "holster",
                         "home theater",
                         "honeycomb",
                         "hook",
                         "hoop skirt",
                         "horizontal bar",
                         "horse-drawn vehicle",
                         "hourglass",
                         "iPod",
                         "clothes iron",
                         "jack-o'-lantern",
                         "jeans",
                         "jeep",
                         "T-shirt",
                         "jigsaw puzzle",
                         "pulled rickshaw",
                         "joystick",
                         "kimono",
                         "knee pad",
                         "knot",
                         "lab coat",
                         "ladle",
                         "lampshade",
                         "laptop computer",
                         "lawn mower",
                         "lens cap",
                         "paper knife",
                         "library",
                         "lifeboat",
                         "lighter",
                         "limousine",
                         "ocean liner",
                         "lipstick",
                         "slip-on shoe",
                         "lotion",
                         "speaker",
                         "loupe",
                         "sawmill",
                         "magnetic compass",
                         "mail bag",
                         "mailbox",
                         "tights",
                         "tank suit",
                         "manhole cover",
                         "maraca",
                         "marimba",
                         "mask",
                         "match",
                         "maypole",
                         "maze",
                         "measuring cup",
                         "medicine chest",
                         "megalith",
                         "microphone",
                         "microwave oven",
                         "military uniform",
                         "milk can",
                         "minibus",
                         "miniskirt",
                         "minivan",
                         "missile",
                         "mitten",
                         "mixing bowl",
                         "mobile home",
                         "Model T",
                         "modem",
                         "monastery",
                         "monitor",
                         "moped",
                         "mortar",
                         "square academic cap",
                         "mosque",
                         "mosquito net",
                         "scooter",
                         "mountain bike",
                         "tent",
                         "computer mouse",
                         "mousetrap",
                         "moving van",
                         "muzzle",
                         "nail",
                         "neck brace",
                         "necklace",
                         "nipple",
                         "notebook computer",
                         "obelisk",
                         "oboe",
                         "ocarina",
                         "odometer",
                         "oil filter",
                         "organ",
                         "oscilloscope",
                         "overskirt",
                         "bullock cart",
                         "oxygen mask",
                         "packet",
                         "paddle",
                         "paddle wheel",
                         "padlock",
                         "paintbrush",
                         "pajamas",
                         "palace",
                         "pan flute",
                         "paper towel",
                         "parachute",
                         "parallel bars",
                         "park bench",
                         "parking meter",
                         "passenger car",
                         "patio",
                         "payphone",
                         "pedestal",
                         "pencil case",
                         "pencil sharpener",
                         "perfume",
                         "Petri dish",
                         "photocopier",
                         "plectrum",
                         "Pickelhaube",
                         "picket fence",
                         "pickup truck",
                         "pier",
                         "piggy bank",
                         "pill bottle",
                         "pillow",
                         "ping-pong ball",
                         "pinwheel",
                         "pirate ship",
                         "pitcher",
                         "hand plane",
                         "planetarium",
                         "plastic bag",
                         "plate rack",
                         "plow",
                         "plunger",
                         "Polaroid camera",
                         "pole",
                         "police van",
                         "poncho",
                         "billiard table",
                         "soda bottle",
                         "pot",
                         "potter's wheel",
                         "power drill",
                         "prayer rug",
                         "printer",
                         "prison",
                         "projectile",
                         "projector",
                         "hockey puck",
                         "punching bag",
                         "purse",
                         "quill",
                         "quilt",
                         "race car",
                         "racket",
                         "radiator",
                         "radio",
                         "radio telescope",
                         "rain barrel",
                         "recreational vehicle",
                         "reel",
                         "reflex camera",
                         "refrigerator",
                         "remote control",
                         "restaurant",
                         "revolver",
                         "rifle",
                         "rocking chair",
                         "rotisserie",
                         "eraser",
                         "rugby ball",
                         "ruler",
                         "running shoe",
                         "safe",
                         "safety pin",
                         "salt shaker",
                         "sandal",
                         "sarong",
                         "saxophone",
                         "scabbard",
                         "weighing scale",
                         "school bus",
                         "schooner",
                         "scoreboard",
                         "CRT screen",
                         "screw",
                         "screwdriver",
                         "seat belt",
                         "sewing machine",
                         "shield",
                         "shoe store",
                         "shoji",
                         "shopping basket",
                         "shopping cart",
                         "shovel",
                         "shower cap",
                         "shower curtain",
                         "ski",
                         "ski mask",
                         "sleeping bag",
                         "slide rule",
                         "sliding door",
                         "slot machine",
                         "snorkel",
                         "snowmobile",
                         "snowplow",
                         "soap dispenser",
                         "soccer ball",
                         "sock",
                         "solar thermal collector",
                         "sombrero",
                         "soup bowl",
                         "space bar",
                         "space heater",
                         "space shuttle",
                         "spatula",
                         "motorboat",
                         "spider web",
                         "spindle",
                         "sports car",
                         "spotlight",
                         "stage",
                         "steam locomotive",
                         "through arch bridge",
                         "steel drum",
                         "stethoscope",
                         "scarf",
                         "stone wall",
                         "stopwatch",
                         "stove",
                         "strainer",
                         "tram",
                         "stretcher",
                         "couch",
                         "stupa",
                         "submarine",
                         "suit",
                         "sundial",
                         "sunglass",
                         "sunglasses",
                         "sunscreen",
                         "suspension bridge",
                         "mop",
                         "sweatshirt",
                         "swimsuit",
                         "swing",
                         "switch",
                         "syringe",
                         "table lamp",
                         "tank",
                         "tape player",
                         "teapot",
                         "teddy bear",
                         "television",
                         "tennis ball",
                         "thatched roof",
                         "front curtain",
                         "thimble",
                         "threshing machine",
                         "throne",
                         "tile roof",
                         "toaster",
                         "tobacco shop",
                         "toilet seat",
                         "torch",
                         "totem pole",
                         "tow truck",
                         "toy store",
                         "tractor",
                         "semi-trailer truck",
                         "tray",
                         "trench coat",
                         "tricycle",
                         "trimaran",
                         "tripod",
                         "triumphal arch",
                         "trolleybus",
                         "trombone",
                         "tub",
                         "turnstile",
                         "typewriter keyboard",
                         "umbrella",
                         "unicycle",
                         "upright piano",
                         "vacuum cleaner",
                         "vase",
                         "vault",
                         "velvet",
                         "vending machine",
                         "vestment",
                         "viaduct",
                         "violin",
                         "volleyball",
                         "waffle iron",
                         "wall clock",
                         "wallet",
                         "wardrobe",
                         "military aircraft",
                         "sink",
                         "washing machine",
                         "water bottle",
                         "water jug",
                         "water tower",
                         "whiskey jug",
                         "whistle",
                         "wig",
                         "window screen",
                         "window shade",
                         "Windsor tie",
                         "wine bottle",
                         "wing",
                         "wok",
                         "wooden spoon",
                         "wool",
                         "split-rail fence",
                         "shipwreck",
                         "yawl",
                         "yurt",
                         "website",
                         "comic book",
                         "crossword",
                         "traffic sign",
                         "traffic light",
                         "dust jacket",
                         "menu",
                         "plate",
                         "guacamole",
                         "consomme",
                         "hot pot",
                         "trifle",
                         "ice cream",
                         "ice pop",
                         "baguette",
                         "bagel",
                         "pretzel",
                         "cheeseburger",
                         "hot dog",
                         "mashed potato",
                         "cabbage",
                         "broccoli",
                         "cauliflower",
                         "zucchini",
                         "spaghetti squash",
                         "acorn squash",
                         "butternut squash",
                         "cucumber",
                         "artichoke",
                         "bell pepper",
                         "cardoon",
                         "mushroom",
                         "Granny Smith",
                         "strawberry",
                         "orange",
                         "lemon",
                         "fig",
                         "pineapple",
                         "banana",
                         "jackfruit",
                         "custard apple",
                         "pomegranate",
                         "hay",
                         "carbonara",
                         "chocolate syrup",
                         "dough",
                         "meatloaf",
                         "pizza",
                         "pot pie",
                         "burrito",
                         "red wine",
                         "espresso",
                         "cup",
                         "eggnog",
                         "alp",
                         "bubble",
                         "cliff",
                         "coral reef",
                         "geyser",
                         "lakeshore",
                         "promontory",
                         "shoal",
                         "seashore",
                         "valley",
                         "volcano",
                         "baseball player",
                         "bridegroom",
                         "scuba diver",
                         "rapeseed",
                         "daisy",
                         "yellow lady's slipper",
                         "corn",
                         "acorn",
                         "rose hip",
                         "horse chestnut seed",
                         "coral fungus",
                         "agaric",
                         "gyromitra",
                         "stinkhorn mushroom",
                         "earth star",
                         "hen-of-the-woods",
                         "bolete",
                         "ear",
                         "toilet paper"]

# DATA_PATH = None  # TODO: Replace with Your Datasets Folder Path

if DATA_PATH is None:
    raise ValueError('Please update your DATA_PATH variable!')


def default_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def arr2arr(arr1, arr2):
    a2a = {}
    for v1, v2 in zip(arr1, arr2):
        if v1 in a2a.keys():
            if a2a[v1] != v2:
                raise ValueError(f'Prompt targets and Regression targets are not synced')
        else:
            a2a[v1] = v2
    return a2a


class UTK_Faces(Dataset):

    def __init__(self, target, split='all',
                 data_path=join(DATA_PATH, 'UTKFace'),
                 ):
        super(UTK_Faces, self).__init__()

        self.split = split
        self.img_paths = np.array(glob.glob(join(data_path, '*')))
        self.filenames = np.array([x.split('/')[-1] for x in self.img_paths])

        target_place = UTK_ATTRS.index(target)
        self.regr_targets = np.array([int(x.split('_')[target_place]) for x in self.filenames]).astype(int)
        self.regr_targets[self.regr_targets >= 100] = 100
        self.prompt_targets = self.regr_targets.astype(str)
        self.prompt_targets[self.prompt_targets == '100'] = '100+'

        self.prompt2regr = arr2arr(self.prompt_targets, self.regr_targets)
        self.prompt2cls, self.cls2prompt, self.cls2regr = {}, {}, {}
        for i, x in enumerate(sorted(np.unique(self.prompt_targets))):
            self.prompt2cls[x] = i
            self.cls2prompt[i] = x
            self.cls2regr[i] = self.prompt2regr[x]
        self.all_labels_names = np.array(list(self.prompt2cls.keys()))
        self.cls_targets = np.array([self.prompt2cls[x] for x in self.prompt_targets])

        # filter by split
        self.split_indices = self._get_split_indices()
        self.img_paths = self.img_paths[self.split_indices]
        self.filenames = self.filenames[self.split_indices]
        self.regr_targets = self.regr_targets[self.split_indices]
        self.prompt_targets = self.prompt_targets[self.split_indices]
        self.cls_targets = self.cls_targets[self.split_indices]

        normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img = self.transform(default_loader(self.img_paths[i]))
        regr_tgt = self.regr_targets[i]
        return img, regr_tgt

    def _get_split_indices(self):
        np.random.seed(0)
        all_indxs = list(np.arange(len(self.img_paths)))
        if self.split == 'all':
            return all_indxs
        train_indxs, test_indxs = train_test_split(all_indxs, test_size=0.25, stratify=self.regr_targets)
        if self.split == 'train':
            return train_indxs
        elif self.split == 'test':
            return test_indxs
        else:
            raise ValueError(f'No such split value as {self.split}')


class transform_NumpytoPIL(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, img: torch.Tensor):
        """
        Args:
            img (torch.Tensor): Tensor image to be converted to numpy.array

        Returns:
            img (numpy.array): numpy image.
        """
        if np.max(img) <= 1:
            img = (img * 255.).astype(np.uint8)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        if img.shape[-1] == 1:
            img = np.tile(img, (1, 1, 3))
        return PIL.Image.fromarray(img)


class Stanford_Cars(Dataset):

    def __init__(self, data_name, label_name, split='train',
                 data_path=join(DATA_PATH, 'npz')
                 ):
        super(Stanford_Cars, self).__init__()

        self.data_name = data_name
        if split == 'train':
            self.data_name = self.data_name.replace('test', 'train')
        elif split == 'test':
            self.data_name = self.data_name.replace('train', 'test')
        else:
            raise ValueError(f'Unknown split request: {split}')
        self.label_name = label_name

        np_path = join(data_path, 'stanford_cars_balanced__x256__train' + '.npz')
        self.imgs, self.regr_targets, self.prompt_targets = self.load_np(np_path)

        self.prompt2regr = arr2arr(self.prompt_targets, self.regr_targets)
        self.prompt2cls, self.cls2prompt, self.cls2regr = {}, {}, {}
        for i, x in enumerate(sorted(np.unique(self.prompt_targets))):
            self.prompt2cls[x] = i
            self.cls2prompt[i] = x
            self.cls2regr[i] = self.prompt2regr[x]
        self.all_labels_names = np.array(list(self.prompt2cls.keys()))
        self.cls_targets = np.array([self.prompt2cls[x] for x in self.prompt_targets])

        normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transform_NumpytoPIL(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize_transform
        ])

    def load_np(self, np_path):
        data = dict(np.load(np_path, allow_pickle=True))
        imgs = data['imgs']
        target_values = data['contents'][:, np.where(data['colnames'] == self.label_name)[0][0]]
        regr_target_values, prompt_target_values = self.transform_raw_label(target_values)
        return imgs, regr_target_values, prompt_target_values

    def transform_raw_label(self, target_labels):
        if 'smallnorb' in self.data_name:
            if self.label_name == 'azimuth':
                # starts from a label of 0-18
                degs_per_part = 360. / 18.
                radian_targets = target_labels.astype(float) * degs_per_part * (np.pi / 180.)
                prompt_targets = radian_targets * (180. / np.pi)
                prompt_targets = np.round(prompt_targets, 0)
                prompt_targets = prompt_targets.astype(str)
                regr_targets = prompt_targets.astype(float)  # radian_targets
            else:
                raise ValueError(f'No supported attribute as {self.label_name} for dataset {self.data_name}')
        elif 'cars3d' in self.data_name:
            if self.label_name == 'azimuth':
                # starts from a label of 0-24
                degs_per_part = 360. / 24.
                radian_targets = target_labels.astype(float) * degs_per_part * (np.pi / 180.)
                prompt_targets = radian_targets * (180. / np.pi)
                prompt_targets = np.round(prompt_targets, 0)
                prompt_targets = prompt_targets.astype(str)
                regr_targets = prompt_targets.astype(float)  # radian_targets
        elif 'stanford_cars' in self.data_name:
            if self.label_name == 'year':
                prompt_targets = target_labels.astype(str)
                # make regression labels starts from 1
                regr_targets = (target_labels - 1990).astype(int)
            else:
                raise ValueError(f'No supported attribute as {self.label_name} for dataset {self.data_name}')
        else:
            raise ValueError(f'No supported dataset like {self.data_name}')
        return regr_targets, prompt_targets

    def __len__(self):
        return len(self.regr_targets)

    def __getitem__(self, i):
        img = self.transform(self.imgs[i])
        regr_tgt = self.regr_targets[i]
        return img, regr_tgt


class ImageNet(Dataset):

    def __init__(self, transform=None, split='train', data_path=join(DATA_PATH, 'imagenet/imagenet')):

        super(ImageNet, self).__init__()
        self.split = split
        if self.split == 'test':
            self.split = 'val'
        self.datadir = join(data_path, self.split)
        normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_transform
            ])
        else:
            self.transform = transform
        self.orig_data = ImageFolder(self.datadir)

        self.regr_targets = np.ones(len(self.orig_data)).astype(int)
        self.prompt2cls, self.cls2prompt, self.cls2regr = {}, {}, {}
        for i, x in enumerate(IMAGENET_CLASS_LABELS):
            self.prompt2cls[x] = i
            self.cls2prompt[i] = x
            self.cls2regr[i] = 1.
        self.all_labels_names = np.array(IMAGENET_CLASS_LABELS).astype(str)
        self.cls_targets = np.array(self.orig_data.targets).astype(int)

    def __len__(self):
        return len(self.orig_data)

    def __getitem__(self, i):
        img, cls = self.orig_data[i]
        img = self.transform(img)
        return img, cls


class CIFAR10(Dataset):

    def __init__(self, split='train'):
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        ds = torchvision.datasets.CIFAR10
        coarse = {}
        self.set = ds(root='data', train=split, download=True, transform=None, **coarse)

        self.all_labels_names = np.array(CIFAR10_CLASS_LABELS).astype(str)
        self.cls_targets = self.set.targets
        self.prompt2cls, self.cls2prompt, self.cls2regr = {}, {}, {}

        for i, x in enumerate(CIFAR10_CLASS_LABELS):
            self.prompt2cls[x] = i
            self.cls2prompt[i] = x
            self.cls2regr[i] = 1.

    def __getitem__(self, i):
        img = Image.fromarray(self.set.data[i])
        img = self.transform(img)
        cls_tgt = self.set.targets[i]
        return img, cls_tgt

    def __len__(self):
        return len(self.set.data)


class Adience(Dataset):

    def __init__(self, split='all',
                 data_path=join(DATA_PATH, 'Adience')
                 ):
        super(Adience, self).__init__()

        self.split = split
        self.full_metadata = pd.read_csv(join(data_path, 'metadata.csv'))
        if split == 'train':
            selected_folds = [0, 1, 2, 3]
        else:
            selected_folds = [4]
        self.metadata = self.full_metadata[self.full_metadata['fold'].isin(selected_folds)]
        # self.img_paths = np.array(sorted(self.img_paths))
        self.img_paths = np.array([join(data_path, x) for x in self.metadata['img_path'].values])
        self.regr_targets = self.metadata['age'].values
        self.prompt_targets = self.regr_targets.astype(str)

        self.prompt2cls, self.cls2prompt, self.cls2regr, self.prompt2regr = {}, {}, {}, {}
        for i, x in enumerate(sorted(np.unique(self.full_metadata['age']))):
            self.prompt2cls[str(x)] = i
            self.cls2prompt[i] = str(x)
            self.cls2regr[i] = x
            self.prompt2regr[str(x)] = x
        self.all_labels_names = np.array(list(self.prompt2cls.keys()))
        self.cls_targets = np.array([self.prompt2cls[x] for x in self.prompt_targets])

        # filter by split

        normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img = self.transform(default_loader(self.img_paths[i]))
        regr_tgt = self.regr_targets[i]
        return img, regr_tgt
