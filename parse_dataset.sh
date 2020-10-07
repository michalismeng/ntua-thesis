# python3 parser.py --dataset "../my-datasets/JSB ../my-datasets/JSB_LAKH ../my-datasets/Nottingham ../my-datasets/maestro-v2.0.0/2018 ../my-datasets/Jazz-MIDIVAE" \
#                   --dataset_name "JSB JSB_LAKH NMD MAESTRO_18 JAZZ_MV" \
#                   --bars_per_segment "4 4 4 4 4" \
#                   --save_path "dataset/"

python3 parser.py --dataset "../my-datasets/parse_test" \
                  --dataset_name "DUMMY" \
                  --bars_per_segment "4" \
                  --save_path "dataset/"