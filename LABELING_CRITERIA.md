# Manual labeling criteria

A thread is `yes` when it discusses **code duplication**, even in a single passing remark. It does not need to end in a dedup diff. The label describes the whole thread, not the term that made it pass the pre-filter.

## `yes` categories

Categories 1 and 2 can occur together, because a thread may have several patches and each one may do something different. Category 3 never occurs together with categories 1 and 2. Category 4 can occur alone or together with any other category.

The diff is also used as a check. In categories 1 and 2, it confirms that the patch does what the text says, and it can reveal a clone refactoring that the text does not even call dedup (e.g. "consolidate into the core"). In category 3, it confirms that no patch in the thread does clone refactoring or preventive reuse. Without a diff, as in a PULL, what the text states is what counts.

**1. Clone refactoring**
The patch consolidates repeated code into a single place: a helper, a macro, a table, a kernel API or common code. Clone size is not considered, so small repetition also counts, such as a local variable introduced to avoid repeating the same expression, or two constants for the same thing unified into one.
- Example: `CADnq5_MGRpNE8Ge_5=w5mQDJivdS2Wt_2LD=dR-mrWH89Yc-0A@mail.gmail.com` (`[PATCH] drm/amdgpu: deduplicate ring preempt ib function`, `sample_review_usp_v2_consolidated.csv`, row 58)
> "The ring preemption function is identical for both gfx_v11_0 and gfx_v12_0. This patch refactors the code by moving the core logic into a generic function inside amdgpu_gfx.c to reduce code duplication"

**2. Preventive reuse**
Code is moved to a common place with the stated purpose of being used by another component, even if no copy exists yet.
- Example: `CAD=FV=XsUbyB4enDobda3eDoTpCqdgVogyC3YWGe9rsjgR1REw@mail.gmail.com` (`[PATCH v3 0/5] Add and enable GPI DMA users`, `sample_review_v2_consolidated.csv`, row 168)
> "GENI_IF_DISABLE_RO is used by geni spi driver as well to check the status if GENI, so move this to common header qcom-geni-se.h"

**3. Discussion about duplication**
No patch in the thread does clone refactoring or preventive reuse, but someone discusses duplication: asks for dedup, suggests reusing something that already exists, questions, defends or accepts a copy, or attributes a bug to copies that diverged (one was fixed and the other was not). A patch that introduces duplication belongs here when someone comments on the copy, and that includes the author stating the copy (e.g. "this driver is based on foo.c", "copied from the v11 implementation", "same as X but for Y").
- Example: `FFF73D592F13FD46B8700F0A279B802F573BC731@ORSMSX114.amr.corp.intel.com` (`[PATCH V3 0/3] iommu: Add support to change default domain of an iommu`, `sample_review_v2_consolidated.csv`, row 190). The author defends a design choice made to avoid duplicating code, and the reviewer asks twice for it to be changed.
> "I passed it as a parameter because it's already done by iommu_group_store_type() (as below) and I thought that I could save from duplicating code by passing it as a parameter."

**4. Self-admitted technical debt (SATD)**
This category can occur alone or together with any of the others (1, 2 or 3).

A code duplication is admitted as technical debt in at least one of these ways:
- **in the code:** a comment in the diff (TODO, FIXME, XXX, HACK) admits the duplication. E.g. `/* FIXME: duplicated from foo.c, should be shared */`;
- **in the discussion:** someone in the thread admits that a duplication is debt. E.g. "copying for now, will dedup later", or "ok, I'll unify this in a follow-up".

The text must name the duplication or the need to share the code. A TODO that only asks to move code ("move this into a common header") does not count.
- Example: no case in the sample yet.

## What is `no`

- **Redundancy:** removing an unnecessary check, call, assignment or variable. Ask: if both copies stayed, could someone change one and forget the other? If the answer is "the second one simply doesn't need to exist", it is redundancy.
- **Duplication that is not code:** a duplicate table entry, a declaration repeated by mistake, a repeated word in a comment, and other senses of "duplicate", "redundant" and "repeated", such as the I2C "repeated start".
- **Relocation without a reuse motive**, including a TODO like "move to common header" that doesn't say why.
- **A PULL where only a commit title mentions dedup.** A PULL is `yes` only if the maintainer's prose states the reuse or dedup purpose.
- **A clone introduced without anyone commenting on it**: not the author, not a reviewer, and no code comment admitting the copy.
- **Dead code left behind by a refactor and removed as a fix**, with no one mentioning duplication (e.g. `20190819143515.21653-3-Bhawanpreet.Lakha@amd.com`: "during a refactor a redundant code that has unknown behaviour was added").

## General rule

Label only from the content in the sample. Evidence that exists only outside it does not count.
