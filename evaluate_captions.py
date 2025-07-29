import requests
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import json
import os

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)

class CaptionEvaluator:
    def __init__(self, api_url="http://127.0.0.1:5000/"):
        self.api_url = api_url
    
    def evaluate_single(self, image_path, ground_truth_caption):
        """
        Evaluate a single image against its ground truth caption
        """
        try:
            with open(image_path, 'rb') as img_file:
                response = requests.post(
                    self.api_url,
                    files={'image': img_file},
                    data={'ground_truth': ground_truth_caption}
                )
            
            if response.status_code != 200:
                return {"error": f"API request failed with status {response.status_code}"}
                
            result = response.json()
            
            if 'evaluation' not in result:
                return {"error": "No evaluation data in response"}
                
            return result['evaluation']
        
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_batch(self, test_cases, output_file=None):
        """
        Evaluate multiple test cases and optionally save results
        """
        results = []
        for img_path, gt_caption in test_cases:
            res = self.evaluate_single(img_path, gt_caption)
            results.append({
                "image": os.path.basename(img_path),
                "ground_truth": gt_caption,
                "result": res
            })
        
        # Calculate summary statistics
        successful = [r for r in results if 'error' not in r['result']]
        if successful:
            avg_token_acc = sum(r['result']['token_accuracy'] for r in successful) / len(successful)
            avg_bleu = sum(r['result']['bleu_score'] for r in successful) / len(successful)
            exact_matches = sum(r['result']['exact_match'] for r in successful)
            name_incorporation = sum(r['result']['name_incorporation'] for r in successful)
        else:
            avg_token_acc = avg_bleu = exact_matches = name_incorporation = 0
        
        summary = {
            "total_cases": len(results),
            "successful_cases": len(successful),
            "avg_token_accuracy": avg_token_acc,
            "avg_bleu_score": avg_bleu,
            "exact_matches": exact_matches,
            "successful_name_incorporation": name_incorporation,
            "error_cases": len(results) - len(successful)
        }
        
        full_results = {
            "summary": summary,
            "detailed_results": results
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(full_results, f, indent=2)
        
        return full_results

def main():
    # Initialize evaluator
    evaluator = CaptionEvaluator()
    
    # Define test cases with ground truth captions
    test_cases = [
        (r"C:\CIP_blockchain\Cip_vscode\static\captioned_images\20250409_005748_virat_dhoni.jpeg", 
         "Virat and Dhoni are standing in a field"),
        
        (r"C:\CIP_blockchain\Cip_vscode\static\captioned_images\20250409_010010_bird.jpeg", 
         "A bird sitting on a tree branch in a morning"),
        
        (r"C:\CIP_blockchain\Cip_vscode\static\captioned_images\20250409_010033_sundar_sachin.jpeg", 
         "Sundar and Sachin Tendulkar are posing to a picture"),
        
        (r"C:\CIP_blockchain\Cip_vscode\static\captioned_images\20250414_005316_kh3.jpeg", 
         "Kamala harris is standing in front of a microphone"),
        
        (r"C:\CIP_blockchain\Cip_vscode\static\captioned_images\20250413_190725_vijay2jpeg.jpeg", 
         "Vijay is standing infront of flowers"),
        
        (r"C:\CIP_blockchain\Cip_vscode\static\captioned_images\20250409_133616_dhoni_jad.jpeg", 
         "Dhoni and Jadeja are with a cup"),
        
        (r"C:\CIP_blockchain\Cip_vscode\static\captioned_images\20250409_005136_bori2.jpg", 
         "Rishi and Bose is standing in front of a river"),
        
        (r"C:\CIP_blockchain\Cip_vscode\static\captioned_images\20250409_005755_viratdhonbluejpeg.jpeg", 
         "Virat and Dhoni are standing a field"),
        
        (r"C:\CIP_blockchain\Cip_vscode\static\captioned_images\20250413_182634_ajith_helmet.jpeg", 
         "Ajith places his helmet in a table"),
        
        (r"C:\CIP_blockchain\Cip_vscode\static\captioned_images\20250409_005136_bori2.jpg", 
         "Rishi and Bose is standing in front of a river")  # Duplicate image
    ]
    
    # Evaluate all test cases
    results = evaluator.evaluate_batch(test_cases, "caption_evaluation_results.json")
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Tested {results['summary']['total_cases']} images")
    print(f"Average Token Accuracy: {results['summary']['avg_token_accuracy']:.2%}")
    print(f"Average BLEU Score: {results['summary']['avg_bleu_score']:.4f}")
    print(f"Exact Matches: {results['summary']['exact_matches']}")
    print(f"Successful Name Incorporation: {results['summary']['successful_name_incorporation']}")
    
    # Print detailed results
    print("\nDetailed Results:")
    for result in results['detailed_results']:
        print(f"\nImage: {result['image']}")
        print(f"Ground Truth: {result['ground_truth']}")
        if 'error' in result['result']:
            print(f"Error: {result['result']['error']}")
        else:
            print(f"Generated Caption: {result['result']['generated_caption']}")
            print(f"Token Accuracy: {result['result']['token_accuracy']:.2%}")
            print(f"BLEU Score: {result['result']['bleu_score']:.4f}")
            print(f"Exact Match: {'Yes' if result['result']['exact_match'] else 'No'}")
            print(f"Name Incorporated: {'Yes' if result['result']['name_incorporation'] else 'No'}")

if __name__ == "__main__":
    main()