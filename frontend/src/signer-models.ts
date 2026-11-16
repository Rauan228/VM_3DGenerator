/** The signers' files; the version is bumped with each rebuild of the avatars, so browsers do not keep a stale GLB. */
export const SIGNER_VERSION = "20261117b";
export const SIGNER_MODELS = {
  male: `/models/tilmind-male.glb?v=${SIGNER_VERSION}`,
  female: `/models/tilmind-female.glb?v=${SIGNER_VERSION}`,
};
