def clean_text(text):
    """
    Filters out noisy lines like headers, footers, and menus.
    Ensures that only substantial text is preserved for context.
    """
    if not text:
        return ""
        
    lines = text.split("\n")
    cleaned = []
    
    # Noise markers to skip
    noise_keywords = [
        "menu", "search", "jobs", "license", "back to top", 
        "skip to content", "close", "socialize", "copyright",
        "all rights reserved", "terms of service", "privacy policy",
        "download", "macos", "android", "windows"
    ]
    
    for line in lines:
        line = line.strip()
        
        # Skip very short lines or lines containing noise keywords
        if len(line) < 50:
            continue
            
        if any(word in line.lower() for word in noise_keywords):
            continue
            
        # Optional: ensure it looks like a sentence (ends with punctuation)
        # if not any(line.endswith(p) for p in [".", "?", "!"]):
        #     continue
            
        cleaned.append(line)
        
    return "\n".join(cleaned)
