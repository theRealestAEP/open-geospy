export const retrievalControlHelp = {
  reference_image: 'Upload the photo you want to search.',
  search_top_k: 'How many nearest results to return in Search mode.',
  locate_top_k: 'How many location families to return in Locate mode. The ORB gallery uses the same count for compared capture images.',
  min_similarity: 'Optional floor for match similarity. Higher values are stricter and may reduce recall.',
  embedding_base: 'Choose which embedding set this action uses.',
  locate_battlefront_mode: 'Play a cinematic map reveal after a successful locate, with a progressive zoom from regional scale down to the matched street.',
  locate_orb_enabled: 'Run ORB local-feature reranking on the top vector candidates before panorama aggregation.',
  locate_orb_top_n: 'How many top vector candidates should be reranked with ORB.',
  locate_orb_weight: 'How strongly the ORB score boosts the vector score during locate reranking.',
  locate_orb_feature_count: 'How many ORB keypoints to extract per image during reranking. Higher values are slower but can preserve more detail.',
  locate_orb_ransac_top_k: 'How many top ORB candidates should be geometry-checked with RANSAC.',
  locate_orb_ignore_bottom_ratio: 'Optional fraction of the lower frame to ignore during ORB feature extraction.',
  locate_sam2_mask_cars: 'Use local SAM 2 masking to remove cars and other road vehicles from the query image and top ORB review candidates before feature extraction.',
  locate_sam2_mask_trees: 'Experimental: use local SAM 2 masking to remove trees and other vegetation from the query image and top ORB review candidates before feature extraction.',
  locate_orb_popup: 'Optionally auto-open the ORB fingerprint popup while comparisons are running.'
};
