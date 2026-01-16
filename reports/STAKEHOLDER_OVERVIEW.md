# Engineering Drawing REV Extraction Project
## Stakeholder Overview

---

## Project Objective

**Goal:** Automatically extract REV (revision) values from 4,500+ engineering drawings for migration to Bluestar (Engineering Data Management System)

**Challenge:** Manual extraction would take weeks; accuracy is critical for maintaining drawing version control

**Solution:** Hybrid AI pipeline combining rule-based extraction with GPT-4 Vision

---

## Current Pipeline Architecture

### Two-Stage Hybrid Approach

**Stage 1: Fast Rule-Based Extraction (PyMuPDF)**
- Analyzes PDF text and positioning for **text-based PDFs**
- Identifies REV values using domain rules
- Handles majority of standard text-based drawings
- **Performance:** ~3 seconds per drawing
- **Cost:** Minimal

**Stage 2: AI Visual Analysis (GPT-4 Vision)**
- **Triggered for TWO scenarios:**
  1. **Image-based PDFs** (~20-30% of drawings) - PyMuPDF cannot extract text from scanned/image drawings
  2. **Suspicious/failed extractions** - Validates ambiguous PyMuPDF results
- Analyzes drawing image like a human would
- Validates against engineering drawing standards
- **Performance:** ~10 seconds per drawing
- **Cost:** ~$0.01 per drawing

### Why Hybrid?
- **Handles both formats:** Text-based (fast) and image-based (AI required)
- **Smart fallback:** GPT-4 validates suspicious cases automatically
- **Cost-Effective:** AI only used when necessary (~$50-70 vs $450 for all drawings)

---

## Key Challenges & Solutions

### Challenge 1: Mixed PDF Formats (Text vs Image)
**Problem:** Engineering drawings exist in two formats
- **Text-based PDFs:** Native digital drawings with extractable text
- **Image-based PDFs:** Scanned/legacy drawings (~20-30% of dataset)
- PyMuPDF can only process text-based PDFs

**Solution:** Automatic format detection
- Attempt PyMuPDF extraction first
- If text extraction fails → route to GPT-4 Vision
- GPT-4 processes image-based PDFs like a human reading the drawing
- Seamless handling of both formats in single pipeline

### Challenge 2: Edge Case Variations
**Problem:** REV values appear in inconsistent formats
- Numeric: `1-0`, `2-0`, `12-0`
- Letters: `A`, `B`, `AA`, `AB`
- Special: `-`, `_` (indicates no revision)
- Invalid patterns: `5-40`, `DE`, bare numbers

**Solution:** Built validation rules that:
- Accept only valid REV formats based on engineering standards
- Flag suspicious values for GPT verification
- Handle rotated drawings (check all 4 corners)

### Challenge 3: Distinguishing Title Block from Revision Table
**Problem:** Drawings contain multiple "REV" references
- **Title Block** (bottom-right): Current revision ✓
- **Revision Table** (top-right): Historical changes ✗
- Grid references, section markers, part numbers ✗

**Solution:** 
- Spatial analysis: Focus on bottom-right title block region
- Contextual validation: Check for nearby identifiers (DWG NO, SCALE, etc.)
- Visual confirmation via GPT when ambiguous

### Challenge 4: Azure API Rate Limits
**Problem:** Parallel processing hits rate limits (60 requests/minute)
- Multiple workers sending simultaneous requests
- 429 errors causing failures and delays

**Solution:** Implemented lightweight throttling
- Workers coordinate to stay under limits
- SDK automatically retries failed requests
- Processing optimized to 2-4 parallel workers

### Challenge 5: Special Characters & Empty Values
**Problem:** Some drawings have no revision or use placeholders
- Dash (`-`), underscores (`_`), or empty fields
- Difficult to distinguish from extraction failures

**Solution:** 
- Cross-validation: When both engines agree on special char → accept
- GPT confirmation: Visual verification for edge cases
- Domain knowledge: Special chars valid in "not applicable" scenarios

---

## Current Status

### Completed
✅ Hybrid pipeline fully operational  
✅ Handles both text-based and image-based PDFs  
✅ Edge case handling implemented  
✅ Rate limiting optimized  
✅ Full dataset processed (4,500 drawings)  
✅ **Validation complete on 200-file sample**

### Validation Results (n=200)
📊 **Overall Accuracy: 95%**
- Strong performance across standard cases
- Identifies gap vs 98% target
- Error analysis reveals improvement opportunities

### Key Findings
- Text-based PDFs: High accuracy
- Image-based PDFs: Variable accuracy (GPT-dependent)
- Edge cases: Some ambiguous results remain
- **Conclusion:** Dual-model enhancement recommended

---

## Next Steps

### 1. Implement Dual-Model Cross-Validation (Week 1-2)
**Decision Made:** Based on 95% accuracy (below 98% target)

**Approach:**
- Add second GPT model (GPT-4o or alternative configuration)
- Cross-validate results when primary model has low confidence
- Implement voting/consensus mechanism
- Focus on image-based PDFs and edge cases

**Expected Impact:**
- Improve accuracy from 95% → 98%+
- Reduce human review from 5% → 2-3%
- Target error types identified in validation

**Trade-offs:**
- Additional cost: ~$50-70 (2x GPT usage on subset)
- Processing time: +10-15 minutes
- Worth it for critical accuracy improvement

### 2. Re-Validate Enhanced Pipeline (Week 2)
- Test dual-model on original 200-file validation set
- Measure accuracy improvement
- Verify error reduction in problematic categories
- Confirm ready for production

### 3. Production Deployment (Week 3-4)
- Process full 4,500 drawing dataset with enhanced pipeline
- Batch upload to Bluestar
- Route remaining low-confidence cases (~2-3%) for manual review
- Integration with migration team
- Documentation handoff

### 4. Post-Deployment (Week 4+)
- Monitor accuracy in production
- Support engineering team with flagged cases
- Document lessons learned
- Template for future document extraction projects

---

## Success Metrics

| Metric | Target | Current Status |
|--------|--------|----------------|
| **Processing Speed** | <1 hour for 4,500 drawings | ✅ ~45 min |
| **Accuracy Rate** | >98% correct extraction | ⚠️ 95% (validated n=200) |
| **Cost Efficiency** | <$100 total | ✅ ~$50-70 projected |
| **Human Review** | <3% requiring manual check | ⚠️ ~5% currently |
| **API Reliability** | <1% failed requests | ✅ SDK handles retries |

**Status:** 95% accuracy strong but below target → Dual-model enhancement recommended

---

## Technical Highlights (For Reference)

**Tech Stack:**
- PyMuPDF (native PDF parsing)
- GPT-4 Vision (Azure OpenAI)
- Python with parallel processing

**Key Innovations:**
- Multi-corner analysis for rotated drawings
- Spatial + contextual validation rules
- Intelligent GPT fallback (cost optimization)
- Domain-specific validation logic

**Scalability:**
- Can process 10,000+ drawings with same approach
- Rate limiting handles API constraints
- Modular design allows easy enhancements

---

## Risk Assessment

| Risk | Impact | Status | Mitigation |
|------|--------|--------|------------|
| **Accuracy below target** | High | ⚠️ **Identified (95% vs 98%)** | Dual-model validation being implemented |
| **Image-based PDF quality** | Medium | 🔄 **Active** | GPT-4 Vision handles, dual-model improves |
| **Edge cases missed** | Medium | ✅ **Managed** | Human review queue for low-confidence results |
| **API rate limits** | Low | ✅ **Resolved** | Throttling implemented, SDK handles retries |
| **Cost overrun** | Low | ✅ **On Track** | ~$50-70 within budget |

**Overall Risk Level: LOW-MEDIUM**
- Accuracy gap identified but solution ready
- Dual-model enhancement addresses validation findings
- Timeline remains on track (additional 1-2 weeks)

---

## Recommendations

### Immediate Action (This Week)
1. **Proceed with dual-model implementation**
   - Validation results (95%) justify enhancement
   - 3% accuracy gain needed to reach target
   - Cost increase acceptable for critical accuracy

2. **Prioritize error categories**
   - Focus on image-based PDF improvement
   - Address edge cases identified in validation
   - Target systematic error patterns

### Near-term (Weeks 2-3)
3. **Re-validate enhanced pipeline**
   - Use same 200-file test set
   - Confirm accuracy improvement to 98%+
   - Measure reduction in human review needs

4. **Finalize production deployment**
   - Process full 4,500 drawing dataset
   - Deliver results to Bluestar migration team
   - Document enhancement for future projects

### Long-term Considerations
- **Template for future extractions:** This dual-model approach is reusable
- **Other metadata fields:** Extend to DWG NO, SCALE, etc.
- **Continuous improvement:** Monitor production accuracy and refine

---

## Questions for Discussion

1. **Validation Timeline:** What sample size do stakeholders need to see before deployment decision?

2. **Accuracy Threshold:** Is 98% acceptable, or do we need 99%+ (would justify dual-model)?

3. **Human Review Process:** How should we handle the ~7% flagged for review?

4. **Deployment Timing:** Any constraints on when results need to be available?

5. **Future Scope:** Interest in extending this approach to other drawing metadata?

---

## Summary

**Current State:** Hybrid pipeline operational, validation complete, accuracy gap identified

**Key Achievement:** Reduced weeks of manual work to ~45 minutes automated processing

**Validation Findings:** 95% accuracy on 200-file sample (strong but below 98% target)

**Decision Made:** Implement dual-model cross-validation to close accuracy gap

**Next Milestone:** Deploy enhanced pipeline and achieve 98%+ accuracy

**Risk Level:** Low - Clear path to target, solution ready to implement

**Timeline:** Production deployment in 3-4 weeks (includes enhancement + re-validation)
